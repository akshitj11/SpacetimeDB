use std::{
    io,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex, Weak,
    },
};

use parking_lot::{ArcRwLockReadGuard, RawRwLock, RwLock, RwLockWriteGuard};
use slab::Slab;
use spacetimedb_lib::bsatn::DecodeError;
use spacetimedb_sats::layout::Size;

use crate::{
    indexes::{PageIndex, PAGE_SIZE},
    page::{self, Page, PageMetadata},
    page_pool::PagePool,
    tiered::{BudgetExceeded, BudgetPermit, ByteBudget},
};

#[cfg(test)]
use crate::var_len::VarLenMembers;

pub type PageFrameReadGuard = ArcRwLockReadGuard<RawRwLock, Box<Page>>;

pub trait PageBackingStore: Send + Sync + 'static {
    /// Load a [Page] by its content hash from backing storage .
    fn load_page(&self, hash: blake3::Hash) -> Result<Box<Page>, PageIoError>;
}

impl PageBackingStore for () {
    fn load_page(&self, _: blake3::Hash) -> Result<Box<Page>, PageIoError> {
        unimplemented!("no page backing store configured")
    }
}

#[derive(Debug, thiserror::Error)]
pub enum PageError {
    #[error("maximum number of pages exceeded")]
    TooManyPages,
    #[error(transparent)]
    MemoryLimitExceeded(#[from] BudgetExceeded),
    #[error(transparent)]
    Page(page::Error),
    #[error("page at index {0:?} is missing")]
    MissingPage(PageIndex),
    #[error("object {0} is missing")]
    MissingObject(blake3::Hash),
    #[error(transparent)]
    Io(#[from] PageIoError),
    #[error("error decoding page from bsatn")]
    Deserialize(DecodeError),
}

#[derive(Debug, thiserror::Error)]
pub enum PageIoError {
    #[error("expected page hash {expected} doesn't match computed page hash {computed}")]
    HashMismatch {
        expected: blake3::Hash,
        computed: blake3::Hash,
    },
    #[error(transparent)]
    Io(#[from] io::Error),
}

#[derive(Clone, Copy)]
pub enum PageEvictionPolicy {
    Evictable,
    NeverEvict,
}

#[derive(Clone)]
pub struct PageSlotHandle {
    slot: Arc<Mutex<PageSlot>>,
}

impl PageSlotHandle {
    pub fn is_resident(&self) -> bool {
        self.slot.lock().unwrap().is_resident()
    }

    pub fn is_absent(&self) -> bool {
        self.slot.lock().unwrap().is_absent()
    }

    pub fn has_space_for_row(&self, fixed_row_size: Size, num_var_len_granules: usize) -> Option<bool> {
        self.slot
            .lock()
            .unwrap()
            .has_space_for_row(fixed_row_size, num_var_len_granules)
    }

    pub fn is_full(&self, fixed_row_size: Size) -> Option<bool> {
        self.slot.lock().unwrap().is_full(fixed_row_size)
    }

    pub fn available_var_len_granules(&self) -> Option<usize> {
        self.slot.lock().unwrap().available_var_len_granules()
    }

    pub fn bytes_used_by_rows(&self, fixed_row_size: Size) -> usize {
        self.slot.lock().unwrap().bytes_used_by_rows(fixed_row_size)
    }

    pub fn metadata(&self, fixed_row_size: Size) -> Option<PageMetadata> {
        self.slot.lock().unwrap().metadata(fixed_row_size)
    }

    pub fn page(&self) -> Option<PageHandle> {
        self.slot.lock().unwrap().page().cloned()
    }

    pub(super) fn free(&self) {
        let mut slot = self.slot.lock().unwrap();
        *slot = PageSlot::Absent;
    }
}

pub enum PageSlot {
    Absent,
    Resident {
        handle: PageHandle,
        state: ResidentPageState,
    },
    NonResident {
        hash: blake3::Hash,
        metadata: PageMetadata,
    },
}

impl PageSlot {
    pub fn is_resident(&self) -> bool {
        matches!(self, Self::Resident { .. })
    }

    pub fn is_absent(&self) -> bool {
        matches!(self, Self::Absent)
    }

    pub fn has_space_for_row(&self, fixed_row_size: Size, num_var_len_granules: usize) -> Option<bool> {
        match self {
            PageSlot::Absent => None,
            PageSlot::Resident { handle, .. } => {
                Some(handle.read().has_space_for_row(fixed_row_size, num_var_len_granules))
            }
            PageSlot::NonResident { metadata, .. } => {
                Some(metadata.has_space_for_row(fixed_row_size, num_var_len_granules))
            }
        }
    }

    pub fn is_full(&self, fixed_row_size: Size) -> Option<bool> {
        match self {
            PageSlot::Absent => None,
            PageSlot::Resident { handle, .. } => Some(handle.read().is_full(fixed_row_size)),
            PageSlot::NonResident { metadata, .. } => Some(metadata.is_full(fixed_row_size)),
        }
    }

    pub fn available_var_len_granules(&self) -> Option<usize> {
        match self {
            PageSlot::Absent => None,
            PageSlot::Resident { handle, .. } => Some(handle.read().available_var_len_granules()),
            PageSlot::NonResident { metadata, .. } => Some(metadata.available_var_len_granules()),
        }
    }

    pub fn bytes_used_by_rows(&self, fixed_row_size: Size) -> usize {
        match self {
            PageSlot::Absent => 0,
            PageSlot::Resident { handle, .. } => handle.read().bytes_used_by_rows(fixed_row_size),
            PageSlot::NonResident { metadata, .. } => metadata.bytes_used_by_rows as _,
        }
    }

    pub fn metadata(&self, fixed_row_size: Size) -> Option<PageMetadata> {
        match self {
            PageSlot::Absent => None,
            PageSlot::Resident { handle, .. } => Some(handle.read().metadata(fixed_row_size)),
            PageSlot::NonResident { metadata, .. } => Some(*metadata),
        }
    }

    pub fn page(&self) -> Option<&PageHandle> {
        match self {
            PageSlot::Absent | PageSlot::NonResident { .. } => None,
            PageSlot::Resident { handle, .. } => Some(handle),
        }
    }
}

#[derive(Clone)]
pub struct PageHandle {
    frame: Arc<PageFrame>,
}

impl PageHandle {
    pub fn read(&self) -> PageFrameReadGuard {
        self.frame.read()
    }

    pub fn with_page_mut<T>(&mut self, f: impl FnOnce(&mut Page) -> T) -> T {
        let mut guard = self.frame.write();
        f(&mut guard)
    }
}

#[derive(Clone, Copy)]
struct PageFrameId(usize);

pub struct PageFrame {
    id: PageFrameId,
    page: Arc<RwLock<Box<Page>>>,
    #[allow(unused)]
    permit: BudgetPermit,
}

impl PageFrame {
    pub fn read(&self) -> PageFrameReadGuard {
        RwLock::read_arc(&self.page)
    }

    fn write(&self) -> RwLockWriteGuard<'_, Box<Page>> {
        self.page.write()
    }
}

pub enum ResidentPageState {
    Clean { hash: Option<blake3::Hash> },
    Dirty { hash: Option<blake3::Hash> },
}
pub struct PageManager {
    frames: RwLock<FrameRegistry>,
    pool: PagePool,
    store: Arc<dyn PageBackingStore>,
    memory: ByteBudget,
    access_epoch: AtomicU64,
}

impl PageManager {
    pub fn new(pool: PagePool, store: Arc<dyn PageBackingStore>, memory: ByteBudget) -> Self {
        Self {
            frames: <_>::default(),
            pool,
            store,
            memory,
            access_epoch: <_>::default(),
        }
    }

    pub fn get(
        &self,
        slot: &PageSlotHandle,
        eviction_policy: PageEvictionPolicy,
    ) -> Result<Option<PageHandle>, PageError> {
        Ok(self.may_fault(slot, eviction_policy)?.map(|frame| PageHandle { frame }))
    }

    pub fn with_page_mut<T>(
        &self,
        slot: &PageSlotHandle,
        eviction_policy: PageEvictionPolicy,
        f: impl FnOnce(&mut Page) -> T,
    ) -> Result<T, PageError> {
        let frame = self
            .may_fault(slot, eviction_policy)?
            .expect("page requested for mutation to be present");
        let res = {
            let mut page = frame.write();
            f(&mut page)
        };
        let mut slot = slot.slot.lock().unwrap();
        *slot = PageSlot::Resident {
            handle: PageHandle { frame },
            state: ResidentPageState::Dirty { hash: None },
        };
        Ok(res)
    }

    fn may_fault(
        &self,
        slot: &PageSlotHandle,
        eviction_policy: PageEvictionPolicy,
    ) -> Result<Option<Arc<PageFrame>>, PageError> {
        let mut slot_guard = slot.slot.lock().unwrap();
        match *slot_guard {
            PageSlot::Absent => Ok(None),
            PageSlot::Resident { ref handle, .. } => {
                let frame = handle.frame.clone();
                self.frames
                    .write()
                    .touch(frame.id, self.access_epoch.fetch_add(1, Ordering::Relaxed));
                Ok(Some(frame))
            }
            PageSlot::NonResident { hash, .. } => {
                let permit = self.acquire_memory_budget_permit()?;
                let page = self.store.load_page(hash)?;
                let frame = self.frames.write().register(permit, page, |frame| {
                    let weak_frame = Arc::downgrade(&frame);
                    let resident = PageSlot::Resident {
                        handle: PageHandle { frame },
                        state: ResidentPageState::Clean { hash: Some(hash) },
                    };
                    *slot_guard = resident;
                    FrameRegistryEntry {
                        frame: weak_frame,
                        slot: Arc::downgrade(&slot.slot),
                        eviction_policy,
                        last_access_epoch: self.access_epoch.fetch_add(1, Ordering::Relaxed),
                        last_access_count: 0,
                    }
                });

                Ok(Some(frame))
            }
        }
    }

    pub fn allocate(
        &self,
        fixed_row_size: Size,
        eviction_policy: PageEvictionPolicy,
    ) -> Result<PageSlotHandle, PageError> {
        let reservation = self.reserve(fixed_row_size)?;
        Ok(self.acquire(reservation, eviction_policy))
    }

    pub fn reserve(&self, fixed_row_size: Size) -> Result<ReservedPage, BudgetExceeded> {
        let permit = self.acquire_memory_budget_permit()?;
        let page = self.pool.take_with_fixed_row_size(fixed_row_size);

        Ok(ReservedPage { permit, page })
    }

    pub fn acquire(
        &self,
        ReservedPage { permit, page }: ReservedPage,
        eviction_policy: PageEvictionPolicy,
    ) -> PageSlotHandle {
        let slot = Arc::new(Mutex::new(PageSlot::Absent));
        {
            let mut slot_guard = slot.lock().unwrap();
            self.frames.write().register(permit, page, |frame| {
                let weak_frame = Arc::downgrade(&frame);
                *slot_guard = PageSlot::Resident {
                    handle: PageHandle { frame },
                    state: ResidentPageState::Clean { hash: None },
                };
                FrameRegistryEntry {
                    frame: weak_frame,
                    slot: Arc::downgrade(&slot),
                    eviction_policy,
                    last_access_epoch: self.access_epoch.fetch_add(1, Ordering::Relaxed),
                    last_access_count: 0,
                }
            });
        }

        PageSlotHandle { slot }
    }

    pub(super) fn register(
        &self,
        eviction_policy: PageEvictionPolicy,
        pages: impl IntoIterator<Item = Option<Box<Page>>>,
    ) -> Vec<PageSlotHandle> {
        let mut handles = Vec::new();
        for page in pages {
            let slot = Arc::new(Mutex::new(PageSlot::Absent));
            if let Some(page) = page {
                let permit = self.force_acquire_memory_budget_limit();
                let mut slot_gard = slot.lock().unwrap();
                self.frames.write().register(permit, page, |frame| {
                    let weak_frame = Arc::downgrade(&frame);
                    *slot_gard = PageSlot::Resident {
                        handle: PageHandle { frame },
                        state: ResidentPageState::Clean { hash: None },
                    };
                    FrameRegistryEntry {
                        frame: weak_frame,
                        slot: Arc::downgrade(&slot),
                        eviction_policy,
                        last_access_epoch: self.access_epoch.fetch_add(1, Ordering::Relaxed),
                        last_access_count: 0,
                    }
                });
            }

            handles.push(PageSlotHandle { slot });
        }

        handles
    }

    fn force_acquire_memory_budget_limit(&self) -> BudgetPermit {
        self.memory.force_acquire(PAGE_SIZE as _)
    }

    fn acquire_memory_budget_permit(&self) -> Result<BudgetPermit, BudgetExceeded> {
        // TODO: Try to evict pages if acquisition fails.
        self.memory.acquire(PAGE_SIZE as _)
    }
}

#[derive(Default)]
struct FrameRegistry {
    frames: Slab<FrameRegistryEntry>,
}

impl FrameRegistry {
    pub fn register(
        &mut self,
        permit: BudgetPermit,
        page: Box<Page>,
        mk_entry: impl FnOnce(Arc<PageFrame>) -> FrameRegistryEntry,
    ) -> Arc<PageFrame> {
        let entry = self.frames.vacant_entry();
        let frame = Arc::new(PageFrame {
            id: PageFrameId(entry.key()),
            permit,
            page: Arc::new(RwLock::new(page)),
        });
        entry.insert(mk_entry(frame.clone()));
        frame
    }

    pub fn touch(&mut self, id: PageFrameId, epoch: u64) {
        let frame = &mut self.frames[id.0];
        frame.last_access_epoch = epoch;
        frame.last_access_count += 1;
    }
}

pub struct FrameRegistryEntry {
    frame: Weak<PageFrame>,
    slot: Weak<Mutex<PageSlot>>,
    eviction_policy: PageEvictionPolicy,
    last_access_epoch: u64,
    last_access_count: u64,
}
