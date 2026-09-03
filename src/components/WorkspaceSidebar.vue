<script setup>
import { computed, reactive, ref, watch, onBeforeUnmount } from 'vue'
import lumenfolioLogo from '../assets/lumenfolio-logo-transparent.png'
import { startWindowDrag } from '../windowDrag'
import { beginDocDrag, deliverDocDrop, endDocDrag } from '../docDrag'

const props = defineProps({
  roots: {
    type: Array,
    default: () => [],
  },
  // Knowledge-base pivot: logical collections (flat list of
  // { id, parentId, name, position }) + the full document pool. When present,
  // the expanded tree groups documents by collection instead of by disk root.
  collections: {
    type: Array,
    default: () => [],
  },
  documents: {
    type: Array,
    default: () => [],
  },
  selectedDocId: {
    type: String,
    required: true,
  },
  selectedDoc: {
    type: Object,
    default: null,
  },
  filter: {
    type: String,
    default: '',
  },
  scanStatus: {
    type: String,
    default: 'idle',
  },
  scanError: {
    type: String,
    default: '',
  },
  collapsed: {
    type: Boolean,
    default: false,
  },
  locale: {
    type: String,
    required: true,
  },
  ui: {
    type: Object,
    required: true,
  },
  dropActive: {
    type: Boolean,
    default: false,
  },
  dropTargetRootId: {
    type: String,
    default: '',
  },
  // Collection id (or '__unfiled__') the OS-file drag is currently hovering.
  dropTargetCollectionId: {
    type: String,
    default: '',
  },
  trendingActive: {
    type: Boolean,
    default: false,
  },
  trendingEnabled: {
    type: Boolean,
    default: true,
  },
  graphActive: {
    type: Boolean,
    default: false,
  },
  graphEnabled: {
    type: Boolean,
    default: true,
  },
  // True when the center shows the "ask my knowledge base" home (no doc open).
  homeActive: {
    type: Boolean,
    default: false,
  },
})

const emit = defineEmits([
  'update:filter',
  'open-trending',
  'open-graph',
  'new-note',
  'select-doc',
  'add-folder',
  'import-files',
  'add-pdfs',
  'rescan',
  'reindex-doc',
  'delete-doc',
  'reveal-doc',
  'open-settings',
  'open-workspace',
  'delete-root',
  'toggle-root',
  'toggle-collapse',
  'workspace-drop',
  'set-drop-active',
  'new-collection',
  'rename-collection',
  'delete-collection',
  'select-collection',
  'move-doc-to-collection',
  'move-collection',
  'reorder-documents',
  'reorder-collections',
  'clear-unfiled',
  'go-home',
])

// ── Internal drag-and-drop (pointer-based) ─────────────────────────────────
// HTML5 drag-and-drop is unusable here. Tauri's native OS file-drop (needed to
// import files dropped from Finder) is enabled, and on macOS WKWebView that
// routes *every* in-webview drag through the native layer too: the `drop` event
// never lands on the element, the cursor shows the OS "copy (+)" affordance, and
// the window's file-drop overlay flashes. So the tree's own three gestures —
// refile (doc INTO a folder), reparent (folder INTO a folder), and reorder
// (BETWEEN sibling rows) — run on plain mouse events, which the native layer
// leaves alone. OS file drops keep using the native path, untouched.

const DRAG_THRESHOLD_PX = 4

// The active pointer-drag. `label` feeds the floating ghost; `x`/`y` track it.
const dragging = reactive({ active: false, kind: '', id: '', label: '', x: 0, y: 0 })

// Where the dragged item would land. `key` matches a row's data-row-key; `mode`
// is 'into' (inside a folder, reuses the drag-over highlight) or 'before'/'after'
// (an insertion line between sibling rows).
const dropHint = reactive({ key: '', mode: '' })

// Captured on mousedown; promoted to a real drag only past the move threshold so
// a plain click still selects.
let pendingDrag = null

function rowKey(row) {
  return row.type === 'doc' ? `doc-${row.doc.id}` : `col-${row.collection.id}`
}

function resetDrag() {
  pendingDrag = null
  dragging.active = false
  dragging.kind = ''
  dragging.id = ''
  dragging.label = ''
  dropHint.key = ''
  dropHint.mode = ''
  endDocDrag()
  document.body.style.userSelect = ''
  window.removeEventListener('mousemove', onDragMouseMove)
  window.removeEventListener('mouseup', onDragMouseUp)
}

function onRowMouseDown(event, row) {
  // Left button only. Ignore presses on the row's own controls (caret, action
  // buttons, rename input) so those keep clicking; the name / body still drags.
  if (event.button !== 0) return
  if (event.target.closest('.collection-caret, .collection-action-btn, .collection-rename-input')) return
  // Stop WebKit from starting a native text selection on this press. Setting
  // user-select once the drag passes the threshold is far too late — the
  // selection has already begun and grows across every pane the pointer crosses.
  event.preventDefault()
  pendingDrag = {
    kind: row.type === 'doc' ? 'doc' : 'collection',
    id: row.type === 'doc' ? row.doc.id : row.collection.id,
    label: row.type === 'doc' ? (row.doc.shortTitle || row.doc.title || '') : (row.collection.name || ''),
    startX: event.clientX,
    startY: event.clientY,
  }
  window.addEventListener('mousemove', onDragMouseMove)
  window.addEventListener('mouseup', onDragMouseUp)
}

function onDragMouseMove(event) {
  if (!pendingDrag) return
  if (!dragging.active) {
    if (Math.hypot(event.clientX - pendingDrag.startX, event.clientY - pendingDrag.startY) < DRAG_THRESHOLD_PX) {
      return
    }
    dragging.active = true
    dragging.kind = pendingDrag.kind
    dragging.id = pendingDrag.id
    dragging.label = pendingDrag.label
    // Publish document drags so drop zones outside this component (the chat
    // composer) can light up as targets.
    if (dragging.kind === 'doc') beginDocDrag(dragging.id, dragging.label)
    document.body.style.userSelect = 'none'
  }
  dragging.x = event.clientX
  dragging.y = event.clientY
  updateDropHint(event.clientX, event.clientY)
}

// Hit-test the row under the pointer and decide where the drop would land.
function updateDropHint(x, y) {
  const rowEl = document.elementFromPoint(x, y)?.closest('[data-row-key]')
  const key = rowEl?.getAttribute('data-row-key') || ''
  const row = key ? treeRows.value.find((candidate) => rowKey(candidate) === key) : null
  if (!row) {
    dropHint.key = ''
    dropHint.mode = ''
    return
  }
  const rect = rowEl.getBoundingClientRect()
  const offset = rect.height > 0 ? (y - rect.top) / rect.height : 0.5
  dropHint.key = key
  dropHint.mode = dropModeFor(row, offset)
}

function dropModeFor(row, offset) {
  if (row.type === 'collection') {
    // A document can only land INSIDE a folder (folders and documents occupy
    // separate zones, so there's no doc-vs-folder ordering). A folder's top and
    // bottom edges reorder among sibling folders; its middle reparents into it.
    return dragging.kind === 'doc' ? 'into' : offset < 0.3 ? 'before' : offset > 0.7 ? 'after' : 'into'
  }
  if (row.type === 'doc' && dragging.kind === 'doc') {
    return offset < 0.5 ? 'before' : 'after'
  }
  return ''
}

function onDragMouseUp(event) {
  const wasDragging = dragging.active
  const kind = dragging.kind
  const draggedId = dragging.id
  const key = dropHint.key
  const mode = dropHint.mode
  const x = event?.clientX ?? dragging.x
  const y = event?.clientY ?? dragging.y
  resetDrag()
  if (!wasDragging) return // a plain click; let it select
  swallowNextClick()
  if (!draggedId) return
  // A document released over a registered zone outside the tree (the chat
  // composer) goes there. Mutually exclusive with a tree drop: no row is under
  // the pointer out there, so dropHint is empty anyway.
  if (kind === 'doc' && deliverDocDrop(x, y, draggedId)) return
  if (!mode || !key) return
  const row = treeRows.value.find((candidate) => rowKey(candidate) === key)
  if (row) performDrop(kind, draggedId, row, mode)
}

// A drag's mouseup synthesizes a click (on the common ancestor of down/up); eat
// exactly that one so a reorder doesn't also select a row. Self-removing, with a
// next-frame fallback so it can never swallow a later, unrelated click.
function swallowNextClick() {
  const swallow = (event) => {
    event.stopPropagation()
    event.preventDefault()
    window.removeEventListener('click', swallow, true)
  }
  window.addEventListener('click', swallow, true)
  requestAnimationFrame(() => window.removeEventListener('click', swallow, true))
}

function performDrop(kind, draggedId, row, mode) {
  if (mode === 'into') {
    if (row.type !== 'collection') return
    if (kind === 'doc') {
      emit('move-doc-to-collection', { docId: draggedId, collectionId: row.collection.id })
    } else if (draggedId !== row.collection.id && !isCollectionInSubtree(row.collection.id, draggedId)) {
      emit('move-collection', { id: draggedId, parentId: row.collection.id })
    }
    return
  }
  if (kind === 'doc' && row.type === 'doc') {
    reorderDocRelativeTo(draggedId, row.doc, mode)
  } else if (kind === 'collection' && row.type === 'collection') {
    reorderCollectionRelativeTo(draggedId, row.collection, mode)
  }
}

function onRowClick(row) {
  if (row.type === 'doc') emit('select-doc', row.doc.id)
}

function onCollectionNameClick(collection) {
  toggleCollectionExpanded(collection.id)
}

// Whether `candidateId` is `ancestorId` itself or inside its subtree — used to
// reject dropping a collection into its own descendant before hitting the
// backend (which would reject the cycle with a raw error).
function isCollectionInSubtree(candidateId, ancestorId) {
  if (candidateId === ancestorId) return true
  const childrenOf = collectionChildren.value
  const seen = new Set()
  const stack = [...(childrenOf.get(ancestorId) || [])]
  while (stack.length) {
    const node = stack.pop()
    if (!node || seen.has(node.id)) continue
    seen.add(node.id)
    if (node.id === candidateId) return true
    stack.push(...(childrenOf.get(node.id) || []))
  }
  return false
}

// Build the target collection's document id list in its new order and emit it.
// A same-collection drop reorders; a cross-collection drop files into the
// target's collection instead (precise cross-collection index is deferred).
function reorderDocRelativeTo(draggedId, targetDoc, mode) {
  const scope = targetDoc.collectionId ?? null
  const dragged = allDocs.value.find((doc) => doc.id === draggedId)
  if (!dragged) return
  if ((dragged.collectionId ?? null) !== scope) {
    emit('move-doc-to-collection', { docId: draggedId, collectionId: scope })
    return
  }
  const orderedIds = (docsByCollection.value.get(scope ?? UNFILED_ID) || [])
    .map((doc) => doc.id)
    .filter((id) => id !== draggedId)
  const targetIndex = orderedIds.indexOf(targetDoc.id)
  if (targetIndex < 0) return
  orderedIds.splice(mode === 'before' ? targetIndex : targetIndex + 1, 0, draggedId)
  emit('reorder-documents', { collectionId: scope, orderedIds })
}

function reorderCollectionRelativeTo(draggedId, targetCollection, mode) {
  const dragged = (props.collections || []).find((collection) => collection.id === draggedId)
  if (!dragged) return
  const scope = targetCollection.parentId ?? null
  if ((dragged.parentId ?? null) !== scope) {
    // Dropped between siblings of a different parent → reparent there (append),
    // unless that would nest a folder inside its own subtree.
    if (scope !== null && isCollectionInSubtree(scope, draggedId)) return
    emit('move-collection', { id: draggedId, parentId: scope })
    return
  }
  const orderedIds = (collectionChildren.value.get(scope) || [])
    .map((collection) => collection.id)
    .filter((id) => id !== draggedId)
  const targetIndex = orderedIds.indexOf(targetCollection.id)
  if (targetIndex < 0) return
  orderedIds.splice(mode === 'before' ? targetIndex : targetIndex + 1, 0, draggedId)
  emit('reorder-collections', { parentId: scope, orderedIds })
}

const normalizedFilter = computed(() => String(props.filter || '').trim().toLowerCase())
const hasWorkspace = computed(() => props.roots.some((root) => Boolean(String(root.path || '').trim())))
const isScanning = computed(() => props.scanStatus === 'choosing' || props.scanStatus === 'scanning')
// Knowledge-base pivot: prefer the flat document pool (props.documents). Fall
// back to the legacy roots→folders→docs derivation so the rail keeps working if
// documents haven't been wired up.
const rootDocs = computed(() => props.roots.flatMap((root) => (
  (root.folders || []).flatMap((folder) => folder.docs || [])
)))
const allDocs = computed(() => (
  props.documents.length ? props.documents : rootDocs.value
))
const visibleRailDocs = computed(() => (
  props.documents.length
    ? props.documents
    : props.roots.flatMap((root) => (
      root.collapsed ? [] : (root.folders || []).flatMap((folder) => folder.docs || [])
    ))
))
const selectedRoot = computed(() => props.roots.find((root) => (
  (root.folders || []).some((folder) => (folder.docs || []).some((doc) => doc.id === props.selectedDocId))
)) || props.roots[0] || null)
const dragDepth = ref(0)

const isDropActive = computed(() => props.dropActive || dragDepth.value > 0)

function parseTextList(text) {
  return String(text || '')
    .split('\n')
    .map((line) => line.trim())
    .filter((line) => {
      if (!line || line.startsWith('#')) return false
      if (line.startsWith('file://')) return true
      return /^[A-Za-z]:[\\/]|^\//.test(line)
    })
}

function hasFileEntries(payload) {
  const dataTransfer = payload
  if (!dataTransfer) return false
  if (dataTransfer.files && dataTransfer.files.length) return true

  const types = Array.from(dataTransfer.types || []).map((type) => String(type))
  if (types.includes('Files') || types.includes('application/x-moz-file')) return true
  if (types.includes('public.file-url')) return true
  if (types.includes('URL')) return true

  if (parseTextList(dataTransfer.getData?.('text/uri-list')).length > 0) return true
  if (parseTextList(dataTransfer.getData?.('text/plain')).length > 0) return true
  if (parseTextList(dataTransfer.getData?.('public.file-url')).length > 0) return true
  if (parseTextList(dataTransfer.getData?.('URL')).length > 0) return true
  if (parseTextList(dataTransfer.getData?.('application/x-moz-file')).length > 0) return true

  const items = Array.from(dataTransfer.items || [])
  return items.some((item) => item && item.kind === 'file')
}

function visibleDocs(docs) {
  if (!normalizedFilter.value) return docs
  return docs.filter((doc) => String(doc.title || '').toLowerCase().includes(normalizedFilter.value))
}

function emitDropActive(nextState) {
  emit('set-drop-active', Boolean(nextState))
}

function parseDroppedFilePaths(dataTransfer) {
  if (!dataTransfer) return []
  const files = Array.from(dataTransfer.files || [])
  const filePaths = files
    .map((file) => file?.path)
    .filter((path) => Boolean(path))
    .map((path) => String(path))

  if (filePaths.length) return filePaths

  const uriListPaths = parseTextList(dataTransfer.getData?.('text/uri-list'))
  if (uriListPaths.length) return uriListPaths

  const plainTextPaths = parseTextList(dataTransfer.getData?.('text/plain'))
  if (plainTextPaths.length) return plainTextPaths

  const publicFileUrlPaths = parseTextList(dataTransfer.getData?.('public.file-url'))
  if (publicFileUrlPaths.length) return publicFileUrlPaths

  const urlPaths = parseTextList(dataTransfer.getData?.('URL'))
  if (urlPaths.length) return urlPaths

  const mozillaUrlPaths = parseTextList(dataTransfer.getData?.('application/x-moz-file'))
  if (mozillaUrlPaths.length) return mozillaUrlPaths

  return []
}

function handleDragEnter(event) {
  if (!event?.dataTransfer) return
  event.preventDefault()
  if (!hasFileEntries(event.dataTransfer)) return
  dragDepth.value += 1
  emitDropActive(true)
}

function handleDragOver(event) {
  if (!event?.dataTransfer) return
  event.preventDefault()
  if (!hasFileEntries(event.dataTransfer)) return
  event.dataTransfer.dropEffect = 'copy'
  if (dragDepth.value <= 0) {
    dragDepth.value = 1
    emitDropActive(true)
  }
}

function handleDragLeave(event) {
  if (!event?.dataTransfer) return
  event.preventDefault()
  dragDepth.value = Math.max(0, dragDepth.value - 1)
  if (dragDepth.value === 0) emitDropActive(false)
}

function handleDrop(event) {
  if (!event?.dataTransfer) return
  event.preventDefault()
  const paths = parseDroppedFilePaths(event.dataTransfer)
  emitDropActive(false)
  dragDepth.value = 0
  if (!paths.length) return
  emit('workspace-drop', paths)
}

function localized(value) {
  if (!value || typeof value !== 'object') return value
  return value[props.locale] || value.en || Object.values(value)[0]
}

function statusLabel(status, doc = null) {
  if (status === 'indexed') return props.ui.statusIndexed
  if (status === 'indexing') {
    const percent = Number(doc?.indexProgress?.percent || 0)
    return percent > 0 && percent < 100
      ? `${props.ui.statusIndexing} ${percent}%`
      : props.ui.statusIndexing
  }
  if (status === 'stale') return props.ui.statusStale
  return status
}

function progressPercent(doc) {
  const percent = Number(doc?.indexProgress?.percent || 0)
  return Number.isFinite(percent) ? Math.max(0, Math.min(100, percent)) : 0
}

function treeLabel(doc) {
  return doc?.treeReady ? props.ui.treeReady : props.ui.treeMissing
}

// Trailing metadata on a document row. The value reuses the PDF byte-size label,
// which is meaningless for notes (a few bytes rounds to "0 KB") — suppress that
// so a fresh note reads as a clean tree item, not "hello · 0 KB".
function docMeta(doc) {
  const value = localized(doc?.lastOpened)
  return value && value !== '0 KB' ? value : ''
}

function docStatusKind(doc) {
  const status = doc?.indexStatus
  if (status === 'indexed') return 'ready'
  if (status === 'stale') return 'failed'
  return 'processing'
}

function showDocStatusDot(doc) {
  // Healthy sources are the default — a green LED on every row steals attention
  // from the title. Keep a mark only when something actually needs it.
  return docStatusKind(doc) === 'failed'
}

function docStatusTitle(doc) {
  const kind = docStatusKind(doc)
  if (kind === 'ready') return props.ui.docStatusReady
  if (kind === 'failed') return props.ui.docStatusFailed
  return props.ui.docStatusProcessing
}

function compactDocLabel(doc) {
  const name = String(doc?.shortTitle || doc?.title || 'PDF').replace(/\.pdf$/i, '')
  const compact = name
    .replace(/[^A-Za-z0-9\u4e00-\u9fa5]+/g, ' ')
    .trim()
    .split(/\s+/)
    .find(Boolean) || name
  return compact.slice(0, 4)
}

function compactDocTitle(doc) {
  const parts = [
    doc?.shortTitle || doc?.title || 'PDF',
    localized(doc?.lastOpened),
    treeLabel(doc),
    statusLabel(doc?.status, doc),
  ].filter(Boolean)
  return parts.join(' · ')
}

function rootTitle(root) {
  const path = String(root?.path || '')
  const name = localized(root?.name) || path || 'Workspace'
  return path ? `${name} · ${path}` : name
}

function triggerDeleteRoot(root, event = null) {
  if (event) {
    event.preventDefault()
    event.stopPropagation()
  }
  emit('delete-root', root)
}

// "Add to library" menu: type-agnostic entry (new note / import files / add
// folder) replacing the old PDF-folder-only "+".
const addMenuOpen = ref(false)

function onAddMenuOutsideClick() {
  closeAddMenu()
}

function toggleAddMenu() {
  if (addMenuOpen.value) {
    closeAddMenu()
    return
  }
  addMenuOpen.value = true
  window.addEventListener('click', onAddMenuOutsideClick)
}

function closeAddMenu() {
  addMenuOpen.value = false
  window.removeEventListener('click', onAddMenuOutsideClick)
}

function chooseAdd(kind) {
  closeAddMenu()
  if (kind === 'note') emit('new-note')
  else if (kind === 'files') emit('import-files')
  else emit('add-folder')
}

// ── Collection tree (knowledge-base pivot) ──────────────────────────────────
// Documents are grouped by collectionId; collections nest by parentId. The tree
// is rendered as a flattened, depth-indented list of rows honoring a local
// expand/collapse Set (top-level collections default to expanded).

const UNFILED_ID = '__unfiled__'
const expandedCollections = reactive(new Set())
// Track which collection ids we've already seeded so newly-created top-level
// collections default to expanded without re-expanding ones the user collapsed.
const seededCollectionIds = new Set()

watch(
  () => props.collections,
  (list) => {
    for (const collection of list || []) {
      if (seededCollectionIds.has(collection.id)) continue
      seededCollectionIds.add(collection.id)
      if (collection.parentId == null) expandedCollections.add(collection.id)
    }
  },
  { immediate: true, deep: true },
)

// Map parentId → children (in position order), for building the flattened tree.
const collectionChildren = computed(() => {
  const byParent = new Map()
  for (const collection of props.collections || []) {
    const key = collection.parentId ?? null
    if (!byParent.has(key)) byParent.set(key, [])
    byParent.get(key).push(collection)
  }
  for (const children of byParent.values()) {
    children.sort((a, b) => {
      const posA = Number(a.position ?? 0)
      const posB = Number(b.position ?? 0)
      if (posA !== posB) return posA - posB
      return String(a.name || '').localeCompare(String(b.name || ''))
    })
  }
  return byParent
})

// Documents keyed by collection id. Docs whose collectionId is null/undefined or
// points at a non-existent collection fall into the Unfiled bucket.
const docsByCollection = computed(() => {
  const known = new Set((props.collections || []).map((c) => c.id))
  const map = new Map()
  for (const doc of allDocs.value) {
    const raw = doc?.collectionId
    const key = raw != null && known.has(raw) ? raw : UNFILED_ID
    if (!map.has(key)) map.set(key, [])
    map.get(key).push(doc)
  }
  // Within a collection, honor the manual drag order (position), falling back to
  // title so equal/backfilled positions stay stable and readable.
  for (const docs of map.values()) {
    docs.sort((a, b) => {
      const pa = Number(a.position ?? 0)
      const pb = Number(b.position ?? 0)
      if (pa !== pb) return pa - pb
      return String(a.shortTitle || a.title || '').localeCompare(String(b.shortTitle || b.title || ''))
    })
  }
  return map
})

function collectionDocCount(id) {
  return (docsByCollection.value.get(id) || []).length
}

function isCollectionExpanded(id) {
  return expandedCollections.has(id)
}

function toggleCollectionExpanded(id) {
  if (expandedCollections.has(id)) expandedCollections.delete(id)
  else expandedCollections.add(id)
}

// Flatten the collection forest into an ordered array of rows. Each collection
// row is followed (when expanded) by its child collections then its documents,
// all depth-indented. The Unfiled bucket is always appended at the bottom.
const treeRows = computed(() => {
  const rows = []
  const childrenOf = collectionChildren.value
  // Guards against a hypothetical parent-id cycle and lets us surface orphans.
  const seen = new Set()

  const pushCollection = (collection, depth) => {
    seen.add(collection.id)
    rows.push({ type: 'collection', depth, collection })
    if (isCollectionExpanded(collection.id)) {
      walk(collection.id, depth + 1)
      for (const doc of visibleDocs(docsByCollection.value.get(collection.id) || [])) {
        rows.push({ type: 'doc', depth: depth + 1, doc })
      }
    }
  }

  const walk = (parentId, depth) => {
    const children = childrenOf.get(parentId ?? null) || []
    for (const collection of children) {
      if (seen.has(collection.id)) continue
      pushCollection(collection, depth)
    }
  }
  walk(null, 0)

  // Defensive: a collection unreachable from the root (dangling parent id, or a
  // cycle) would otherwise never render — and its documents, bucketed under it
  // by docsByCollection, would vanish. Surface any such orphan at the top level.
  for (const collection of props.collections || []) {
    if (!seen.has(collection.id)) {
      pushCollection(collection, 0)
    }
  }

  // Loose documents (collection_id NULL) render at the tree root, after the
  // collections — the Obsidian model, where the vault root holds folders and
  // loose notes side by side. There is no synthetic "Unfiled" folder to nest
  // them under, so a brand-new note lands at the root ("outside"), not one level
  // deep. A doc leaves the root by being filed into a collection; it returns by
  // "Remove from collection".
  for (const doc of visibleDocs(docsByCollection.value.get(UNFILED_ID) || [])) {
    rows.push({ type: 'doc', depth: 0, doc, loose: true })
  }
  return rows
})

function rowIndentStyle(depth) {
  return { paddingLeft: `${depth * 14}px` }
}

// Inline rename: which collection row is being edited + its draft text.
const renamingCollectionId = ref('')
const renameDraft = ref('')

function startRenameCollection(collection, event = null) {
  if (event) {
    event.preventDefault()
    event.stopPropagation()
  }
  renamingCollectionId.value = collection.id
  renameDraft.value = collection.name || ''
}

function commitRenameCollection() {
  const id = renamingCollectionId.value
  const name = String(renameDraft.value || '').trim()
  renamingCollectionId.value = ''
  if (id && name) emit('rename-collection', { id, name })
}

function cancelRenameCollection() {
  renamingCollectionId.value = ''
  renameDraft.value = ''
}

// Autofocus + select the inline rename input when it mounts.
const vFocus = {
  mounted(el) {
    el.focus()
    el.select?.()
  },
}

function triggerNewSubcollection(collection, event = null) {
  if (event) {
    event.preventDefault()
    event.stopPropagation()
  }
  // Ensure the parent is expanded so the new child is visible.
  expandedCollections.add(collection.id)
  emit('new-collection', collection.id)
}

// Inline two-step delete confirm. `window.confirm` is unreliable in the Tauri
// webview (returns false on WKWebView), so the row swaps its actions for a
// ✓/✗ pair instead of popping a native dialog. Delete is non-destructive:
// sources drop to Unfiled, files on disk are untouched.
const confirmDeleteId = ref(null)

function requestDeleteCollection(collection, event = null) {
  if (event) {
    event.preventDefault()
    event.stopPropagation()
  }
  confirmDeleteId.value = collection.id
}

function confirmDeleteCollection(collection, event = null) {
  if (event) {
    event.preventDefault()
    event.stopPropagation()
  }
  confirmDeleteId.value = null
  emit('delete-collection', collection.id)
}

function cancelDeleteCollection(event = null) {
  if (event) {
    event.preventDefault()
    event.stopPropagation()
  }
  confirmDeleteId.value = null
}

// Obsidian-style right-click menu. `kind`: 'blank' (empty tree area → top level)
// or 'collection' (a folder row → scoped to it).
const contextMenu = reactive({
  open: false,
  x: 0,
  y: 0,
  kind: 'blank',
  collection: null,
  doc: null,
})

function openContextMenu(event, kind, collection = null, doc = null) {
  event.preventDefault()
  event.stopPropagation()
  contextMenu.open = true
  contextMenu.x = event.clientX
  contextMenu.y = event.clientY
  contextMenu.kind = kind
  contextMenu.collection = collection
  contextMenu.doc = doc
  // Dedupe: a second right-click while open must not stack listeners.
  window.removeEventListener('click', closeContextMenu)
  window.addEventListener('click', closeContextMenu)
}

function closeContextMenu() {
  if (!contextMenu.open) return
  contextMenu.open = false
  contextMenu.collection = null
  contextMenu.doc = null
  window.removeEventListener('click', closeContextMenu)
}

function ctxReindexDoc() {
  const doc = contextMenu.doc
  closeContextMenu()
  if (doc) emit('reindex-doc', doc)
}

function ctxDeleteDoc() {
  const doc = contextMenu.doc
  closeContextMenu()
  if (doc) emit('delete-doc', doc)
}

// Reveal the source's file in Finder / Explorer — imports reference a file
// where it already lives, so this is how the user finds it again.
function ctxRevealDoc() {
  const doc = contextMenu.doc
  closeContextMenu()
  if (doc) emit('reveal-doc', doc)
}

// Un-file a document: move it out of its collection back to the tree root
// (collection_id → null). The inverse of dragging it into a collection.
function ctxUnfileDoc() {
  const doc = contextMenu.doc
  closeContextMenu()
  if (doc) emit('move-doc-to-collection', { docId: doc.id, collectionId: null })
}

function ctxNewNote() {
  // Filing a new note into a collection: select it first, then create (the
  // parent reads the selected collection at create time).
  if (contextMenu.collection) emit('select-collection', contextMenu.collection.id)
  closeContextMenu()
  emit('new-note')
}

function ctxNewCollection() {
  const parentId = contextMenu.collection ? contextMenu.collection.id : null
  closeContextMenu()
  emit('new-collection', parentId)
}

function ctxImportFiles() {
  if (contextMenu.collection) emit('select-collection', contextMenu.collection.id)
  closeContextMenu()
  emit('import-files')
}

function ctxRenameCollection() {
  const collection = contextMenu.collection
  closeContextMenu()
  if (collection) startRenameCollection(collection)
}

function ctxDeleteCollection() {
  const collection = contextMenu.collection
  closeContextMenu()
  if (collection) requestDeleteCollection(collection)
}

onBeforeUnmount(() => {
  window.removeEventListener('click', onAddMenuOutsideClick)
  window.removeEventListener('click', closeContextMenu)
  // A drag mid-flight leaves window listeners + a userSelect lock behind.
  resetDrag()
})

</script>

<template>
  <aside
    class="sidebar"
    :class="{ collapsed, 'drag-active': isDropActive }"
    @dragenter="handleDragEnter"
    @dragover="handleDragOver"
    @dragleave="handleDragLeave"
    @drop="handleDrop"
  >
    <div class="sidebar-window-bar" data-tauri-drag-region @mousedown="startWindowDrag">
    </div>

    <template v-if="collapsed">
      <div class="rail-brand" title="Lumenfolio" data-tauri-drag-region @mousedown="startWindowDrag">
        <img :src="lumenfolioLogo" alt="" />
      </div>

      <!-- Knowledge-base pivot: the collapsed rail mirrors the expanded section
           ribbon (one mental model). Sources expands the sidebar; the rest reuse
           the existing navigation. -->
      <nav class="sidebar-rail-strip collapsed-strip" aria-label="Knowledge base sections">
        <button
          type="button"
          class="rail-mode"
          :class="{ active: homeActive }"
          :title="ui.myKnowledgeBase || ui.sources || 'Home'"
          :aria-label="ui.myKnowledgeBase || ui.sources || 'Home'"
          @click="emit('go-home'); emit('toggle-collapse')"
        ><span aria-hidden="true"><svg viewBox="0 0 24 24" width="1em" height="1em" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 6.5C10.5 5.5 8.4 5 6 5c-1.2 0-2 .3-2 .9v10.7c0 .5.7.8 2 .8 2.4 0 4.5.6 6 1.6M12 6.5c1.5-1 3.6-1.5 6-1.5 1.2 0 2 .3 2 .9v10.7c0 .5-.7.8-2 .8-2.4 0-4.5.6-6 1.6M12 6.5V20" /></svg></span></button>
        <button
          v-if="graphEnabled"
          type="button"
          class="rail-mode"
          :class="{ active: graphActive }"
          :title="ui.knowledgeGraph"
          :aria-label="ui.knowledgeGraph"
          @click="emit('open-graph')"
        ><span aria-hidden="true"><svg viewBox="0 0 24 24" width="1em" height="1em" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="6" cy="7" r="2.1" /><circle cx="18" cy="6.5" r="2.1" /><circle cx="12" cy="18" r="2.1" /><path d="M8 6.9 16 6.6M7.2 8.7 10.9 16.3M16.9 8.1 13.1 16.2" /></svg></span></button>
      </nav>

      <div class="rail-divider"></div>

      <div class="rail-docs" :aria-label="ui.sources">
        <button
          v-for="doc in visibleRailDocs"
          :key="doc.id"
          type="button"
          class="rail-doc"
          :class="{ active: doc.id === selectedDocId && !trendingActive }"
          :title="compactDocTitle(doc)"
          :aria-label="compactDocTitle(doc)"
          @click="emit('select-doc', doc.id)"
        >
          <span class="rail-doc-icon" aria-hidden="true"></span>
          <span class="rail-doc-name">{{ compactDocLabel(doc) }}</span>
          <span v-if="showDocStatusDot(doc)" class="rail-doc-status" :title="docStatusTitle(doc)">
            <span class="doc-status-dot" :class="docStatusKind(doc)"></span>
          </span>
          <span v-if="doc.indexStatus === 'indexing'" class="rail-doc-progress" aria-hidden="true">
            <span :style="{ height: `${progressPercent(doc)}%` }"></span>
          </span>
        </button>
      </div>

      <nav class="rail-actions" aria-label="Lumenfolio actions">
        <div class="add-menu-wrap">
          <button
            type="button"
            class="rail-btn"
            :title="ui.addToLibrary || 'Add'"
            :aria-label="ui.addToLibrary || 'Add'"
            :aria-haspopup="true"
            :aria-expanded="addMenuOpen"
            :disabled="isScanning"
            @mousedown.stop
            @click.stop="toggleAddMenu"
          >
            +
          </button>
          <div v-if="addMenuOpen" class="add-menu collapsed-add-menu" @mousedown.stop>
            <button type="button" class="add-menu-item" @click="chooseAdd('note')">
              <span class="add-menu-ic" aria-hidden="true"><svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M13 3H6.5A1.5 1.5 0 0 0 5 4.5v15A1.5 1.5 0 0 0 6.5 21h11a1.5 1.5 0 0 0 1.5-1.5V9z" /><path d="M13 3v6h6" /><path d="M8.5 13.5h7M8.5 16.5h4.5" /></svg></span>{{ ui.newNote }}
            </button>
            <button type="button" class="add-menu-item" @click="chooseAdd('files')">
              <span class="add-menu-ic" aria-hidden="true"><svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M5 15v2.5A1.5 1.5 0 0 0 6.5 19h11a1.5 1.5 0 0 0 1.5-1.5V15" /><path d="M12 4v10" /><path d="M8 10.5l4 4 4-4" /></svg></span>{{ ui.importFiles || 'Import files…' }}
            </button>
            <button type="button" class="add-menu-item" @click="chooseAdd('folder')">
              <span class="add-menu-ic" aria-hidden="true"><svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M4 6.5A1.5 1.5 0 0 1 5.5 5h3.8a1.5 1.5 0 0 1 1.06.44l1.2 1.12a1.5 1.5 0 0 0 1.06.44H18.5A1.5 1.5 0 0 1 20 8.5V17a1.5 1.5 0 0 1-1.5 1.5h-13A1.5 1.5 0 0 1 4 17z" /></svg></span>{{ ui.addFolder }}
            </button>
          </div>
        </div>
        <button
          v-if="trendingEnabled"
          type="button"
          class="rail-btn rail-btn-muted"
          :class="{ active: trendingActive }"
          :title="ui.trendingPapers"
          :aria-label="ui.trendingPapers"
          @click="emit('open-trending')"
        >
          <svg viewBox="0 0 24 24" width="1em" height="1em" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 3c.4 2.3-1 3.6-2.2 4.9C8.5 9.2 7.5 10.7 7.5 13a4.5 4.5 0 0 0 9 0c0-1.7-.7-3-1.6-4.1-.3.9-1 1.4-1.8 1.5.9-2.4-.4-5.2-1.1-7.4z" /></svg>
        </button>
        <button
          type="button"
          class="rail-btn"
          :title="ui.settings"
          :aria-label="ui.settings"
          @click="emit('open-settings')"
        >
          <svg viewBox="0 0 24 24" width="1em" height="1em" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="3" /><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" /></svg>
        </button>
      </nav>
    </template>

    <template v-else>
    <div class="sidebar-expanded">
      <!-- Knowledge-base pivot (P1): Obsidian-style icon rail. Sources (the tree)
           is the default panel; the other icons reuse the existing navigation. -->
      <nav class="sidebar-rail-strip" aria-label="Knowledge base sections">
        <button
          type="button"
          class="rail-mode"
          :class="{ active: homeActive }"
          :title="ui.myKnowledgeBase || ui.sources || 'Home'"
          :aria-label="ui.myKnowledgeBase || ui.sources || 'Home'"
          @click="emit('go-home')"
        >
          <span aria-hidden="true"><svg viewBox="0 0 24 24" width="1em" height="1em" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 6.5C10.5 5.5 8.4 5 6 5c-1.2 0-2 .3-2 .9v10.7c0 .5.7.8 2 .8 2.4 0 4.5.6 6 1.6M12 6.5c1.5-1 3.6-1.5 6-1.5 1.2 0 2 .3 2 .9v10.7c0 .5-.7.8-2 .8-2.4 0-4.5.6-6 1.6M12 6.5V20" /></svg></span>
        </button>
        <button
          v-if="graphEnabled"
          type="button"
          class="rail-mode"
          :class="{ active: graphActive }"
          :title="ui.knowledgeGraph"
          :aria-label="ui.knowledgeGraph"
          @click="emit('open-graph')"
        ><span aria-hidden="true"><svg viewBox="0 0 24 24" width="1em" height="1em" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="6" cy="7" r="2.1" /><circle cx="18" cy="6.5" r="2.1" /><circle cx="12" cy="18" r="2.1" /><path d="M8 6.9 16 6.6M7.2 8.7 10.9 16.3M16.9 8.1 13.1 16.2" /></svg></span></button>
        <span class="rail-spacer"></span>
        <!-- Trending is a secondary discovery utility, grouped with Settings at the
             bottom — not a primary knowledge-base action. -->
        <button
          v-if="trendingEnabled"
          type="button"
          class="rail-mode rail-mode-muted"
          :class="{ active: trendingActive }"
          :title="ui.trendingPapers"
          :aria-label="ui.trendingPapers"
          @click="emit('open-trending')"
        ><span aria-hidden="true"><svg viewBox="0 0 24 24" width="1em" height="1em" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3c.4 2.3-1 3.6-2.2 4.9C8.5 9.2 7.5 10.7 7.5 13a4.5 4.5 0 0 0 9 0c0-1.7-.7-3-1.6-4.1-.3.9-1 1.4-1.8 1.5.9-2.4-.4-5.2-1.1-7.4z" /></svg></span></button>
        <button type="button" class="rail-mode" :title="ui.settings" :aria-label="ui.settings" @click="emit('open-settings')">
          <span aria-hidden="true"><svg viewBox="0 0 24 24" width="1em" height="1em" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3" /><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" /></svg></span>
        </button>
      </nav>

      <div class="sidebar-main">
    <div class="sidebar-header" data-tauri-drag-region @mousedown="startWindowDrag">
      <span class="panel-label">{{ ui.libraryTitle || 'Library' }}</span>
      <span class="panel-hint">{{ ui.rightClickHint || '' }}</span>
    </div>

    <label class="search-box">
      <span class="search-icon" aria-hidden="true">
        <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="1.8">
          <circle cx="11" cy="11" r="7" />
          <path d="M20 20l-4.3-4.3" stroke-linecap="round" />
        </svg>
      </span>
      <input
        :value="filter"
        type="text"
        :placeholder="ui.searchPlaceholder"
        @input="emit('update:filter', $event.target.value)"
      />
    </label>

    <div class="tree-area" @contextmenu="openContextMenu($event, 'blank')">
      <!-- Knowledge-base pivot: collection tree. Flattened, depth-indented rows
           mixing collection folders and their documents; loose documents (no
           collection) sit at the tree root after the folders. Right-click
           (blank area or a folder) opens the create/organize menu. -->
      <div class="collection-tree">
        <template v-for="row in treeRows" :key="row.type === 'doc' ? `doc-${row.doc.id}` : `col-${row.collection.id}`">
          <!-- Collection node -->
          <div
            v-if="row.type === 'collection'"
            class="collection-row"
            :class="{
              'drag-over': (dropHint.key === `col-${row.collection.id}` && dropHint.mode === 'into') || dropTargetCollectionId === row.collection.id,
              'drop-before': dropHint.key === `col-${row.collection.id}` && dropHint.mode === 'before',
              'drop-after': dropHint.key === `col-${row.collection.id}` && dropHint.mode === 'after',
            }"
            :style="rowIndentStyle(row.depth)"
            @contextmenu.stop="openContextMenu($event, 'collection', row.collection)"
            :data-collection-id="row.collection.id"
            :data-row-key="`col-${row.collection.id}`"
            @mousedown="renamingCollectionId === row.collection.id ? null : onRowMouseDown($event, row)"
          >
            <button
              type="button"
              class="collection-caret"
              :class="{ expanded: isCollectionExpanded(row.collection.id) }"
              :aria-label="isCollectionExpanded(row.collection.id) ? ui.collapse : ui.expand"
              @click.stop="toggleCollectionExpanded(row.collection.id)"
            >
              <svg viewBox="0 0 24 24" width="11" height="11" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M9 6l6 6-6 6" /></svg>
            </button>
            <template v-if="renamingCollectionId === row.collection.id">
              <input
                v-model="renameDraft"
                class="collection-rename-input"
                type="text"
                @click.stop
                @keydown.enter.prevent="commitRenameCollection"
                @keydown.esc.prevent="cancelRenameCollection"
                @blur="commitRenameCollection"
                v-focus
              />
            </template>
            <template v-else>
              <button
                type="button"
                class="collection-name-btn"
                :title="row.collection.name"
                @click="onCollectionNameClick(row.collection)"
              >
                <span class="collection-name">{{ row.collection.name }}</span>
              </button>
              <span class="collection-count">{{ collectionDocCount(row.collection.id) }}</span>
              <span v-if="confirmDeleteId === row.collection.id" class="collection-actions collection-confirm">
                <button
                  type="button"
                  class="collection-action-btn collection-confirm-yes"
                  :title="ui.deleteCollectionConfirm || 'Delete? Sources move to Unfiled.'"
                  :aria-label="ui.deleteCollection || 'Delete'"
                  @click.stop="confirmDeleteCollection(row.collection, $event)"
                ><svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M5 12l5 5L19 7" /></svg></button>
                <button
                  type="button"
                  class="collection-action-btn"
                  :title="ui.cancel || 'Cancel'"
                  :aria-label="ui.cancel || 'Cancel'"
                  @click.stop="cancelDeleteCollection($event)"
                ><svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" aria-hidden="true"><path d="M6 6l12 12M18 6L6 18" /></svg></button>
              </span>
              <span v-else class="collection-actions">
                <button
                  type="button"
                  class="collection-action-btn"
                  :title="ui.newSubcollection || 'New sub-collection'"
                  :aria-label="ui.newSubcollection || 'New sub-collection'"
                  @click.stop="triggerNewSubcollection(row.collection, $event)"
                ><svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><path d="M12 5.5v13M5.5 12h13" /></svg></button>
                <button
                  type="button"
                  class="collection-action-btn"
                  :title="ui.renameCollection || 'Rename'"
                  :aria-label="ui.renameCollection || 'Rename'"
                  @click.stop="startRenameCollection(row.collection, $event)"
                ><svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M4 20h4L18.5 9.5a2.1 2.1 0 0 0-3-3L5 17v3z" /><path d="M13.5 6.5l3 3" /></svg></button>
                <button
                  type="button"
                  class="collection-action-btn collection-delete-btn"
                  :title="ui.deleteCollection || 'Delete'"
                  :aria-label="ui.deleteCollection || 'Delete'"
                  @click.stop="requestDeleteCollection(row.collection, $event)"
                ><svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M4 7h16M7 7l.8 11a2 2 0 0 0 2 1.9h4.4a2 2 0 0 0 2-1.9L17 7M9 7V5.5A1.5 1.5 0 0 1 10.5 4h3A1.5 1.5 0 0 1 15 5.5V7" /></svg></button>
              </span>
            </template>
          </div>

          <!-- Document row (indented under its collection). A <div role=button>
               so the whole row is a pointer-drag handle; dragging is mouse-based
               (see the drag-and-drop block) because HTML5 DnD is intercepted by
               Tauri's native file-drop on macOS. -->
          <div
            v-else
            class="doc-row"
            role="button"
            tabindex="0"
            :class="{
              active: row.doc.id === selectedDocId && !trendingActive,
              'drop-before': dropHint.key === `doc-${row.doc.id}` && dropHint.mode === 'before',
              'drop-after': dropHint.key === `doc-${row.doc.id}` && dropHint.mode === 'after',
            }"
            :style="rowIndentStyle(row.depth)"
            :title="compactDocTitle(row.doc)"
            :data-row-key="`doc-${row.doc.id}`"
            @mousedown="onRowMouseDown($event, row)"
            @click="onRowClick(row)"
            @keydown.enter.self="emit('select-doc', row.doc.id)"
            @contextmenu="openContextMenu($event, 'doc', null, row.doc)"
          >
            <div class="doc-main">
              <span class="doc-name-wrap">
                <span
                  v-if="showDocStatusDot(row.doc)"
                  class="doc-status-dot"
                  :class="docStatusKind(row.doc)"
                  :title="docStatusTitle(row.doc)"
                  :aria-label="docStatusTitle(row.doc)"
                ></span>
                <span class="doc-name">{{ row.doc.shortTitle }}</span>
              </span>
              <span v-if="docMeta(row.doc)" class="doc-time">{{ docMeta(row.doc) }}</span>
            </div>
            <div v-if="row.doc.indexStatus === 'indexing'" class="doc-progress" aria-hidden="true">
              <span :style="{ width: `${progressPercent(row.doc)}%` }"></span>
            </div>
          </div>
        </template>
      </div>

      <div v-if="!allDocs.length && !collections.length" class="empty-tree">
        {{ ui.noSourcesFound }}
      </div>
    </div>

    <div v-if="scanError" class="scan-error">{{ scanError }}</div>

    <div class="sidebar-status">
      <span v-if="scanStatus === 'scanning'">{{ ui.scanningWorkspace }}</span>
      <span v-else-if="allDocs.length">{{ allDocs.length }} {{ ui.sourcesCountLabel || 'sources' }}</span>
    </div>
      </div>
    </div>

    <!-- Obsidian-style right-click menu (create / organize). -->
    <div
      v-if="contextMenu.open"
      class="ctx-menu"
      :style="{ left: `${contextMenu.x}px`, top: `${contextMenu.y}px` }"
      @click.stop
      @contextmenu.prevent.stop
    >
      <!-- Create actions belong to the tree, not to a single source: right-clicking a
           document offers actions for that document instead. -->
      <template v-if="contextMenu.kind !== 'doc'">
        <button type="button" class="ctx-item" @click="ctxNewNote">
          <span class="ctx-ic" aria-hidden="true"><svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M13 3H6.5A1.5 1.5 0 0 0 5 4.5v15A1.5 1.5 0 0 0 6.5 21h11a1.5 1.5 0 0 0 1.5-1.5V9z" /><path d="M13 3v6h6" /><path d="M8.5 13.5h7M8.5 16.5h4.5" /></svg></span>{{ ui.newNote }}
        </button>
        <button type="button" class="ctx-item" @click="ctxNewCollection">
          <span class="ctx-ic" aria-hidden="true"><svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M4 6.5A1.5 1.5 0 0 1 5.5 5h3.8a1.5 1.5 0 0 1 1.06.44l1.2 1.12a1.5 1.5 0 0 0 1.06.44H18.5A1.5 1.5 0 0 1 20 8.5V17a1.5 1.5 0 0 1-1.5 1.5h-13A1.5 1.5 0 0 1 4 17z" /></svg></span>{{ contextMenu.kind === 'collection' ? (ui.newSubcollection || 'New sub-collection') : (ui.newCollection || 'New collection') }}
        </button>
        <button type="button" class="ctx-item" @click="ctxImportFiles">
          <span class="ctx-ic" aria-hidden="true"><svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M5 15v2.5A1.5 1.5 0 0 0 6.5 19h11a1.5 1.5 0 0 0 1.5-1.5V15" /><path d="M12 4v10" /><path d="M8 10.5l4 4 4-4" /></svg></span>{{ ui.importFiles || 'Import files…' }}
        </button>
      </template>
      <template v-if="contextMenu.kind === 'collection'">
        <div class="ctx-sep"></div>
        <button type="button" class="ctx-item" @click="ctxRenameCollection">
          <span class="ctx-ic" aria-hidden="true"><svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M4 20h4L18.5 9.5a2.1 2.1 0 0 0-3-3L5 17v3z" /><path d="M13.5 6.5l3 3" /></svg></span>{{ ui.renameCollection || 'Rename' }}
        </button>
        <button type="button" class="ctx-item ctx-danger" @click="ctxDeleteCollection">
          <span class="ctx-ic" aria-hidden="true"><svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M4 7h16M7 7l.8 11a2 2 0 0 0 2 1.9h4.4a2 2 0 0 0 2-1.9L17 7M9 7V5.5A1.5 1.5 0 0 1 10.5 4h3A1.5 1.5 0 0 1 15 5.5V7" /></svg></span>{{ ui.deleteCollection || 'Delete' }}
        </button>
      </template>
      <template v-else-if="contextMenu.kind === 'doc'">
        <button type="button" class="ctx-item" @click="ctxReindexDoc">
          <span class="ctx-ic" aria-hidden="true"><svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M19.5 12a7.5 7.5 0 1 1-2.2-5.3" /><path d="M19.5 4.5V9H15" /></svg></span>{{ ui.reindexDocument }}
        </button>
        <button type="button" class="ctx-item" @click="ctxRevealDoc">
          <span class="ctx-ic" aria-hidden="true"><svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M4 6.5A1.5 1.5 0 0 1 5.5 5h3.8a1.5 1.5 0 0 1 1.06.44l1.2 1.12a1.5 1.5 0 0 0 1.06.44H18.5A1.5 1.5 0 0 1 20 8.5V17a1.5 1.5 0 0 1-1.5 1.5h-13A1.5 1.5 0 0 1 4 17z" /><circle cx="12" cy="12.5" r="2.2" /><path d="M13.7 14.2 15.5 16" /></svg></span>{{ ui.revealInFileManager }}
        </button>
        <button
          v-if="contextMenu.doc && contextMenu.doc.collectionId"
          type="button"
          class="ctx-item"
          @click="ctxUnfileDoc"
        >
          <span class="ctx-ic" aria-hidden="true"><svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3v10" /><path d="M8 7l4-4 4 4" /><path d="M5 15v3a1.5 1.5 0 0 0 1.5 1.5h11A1.5 1.5 0 0 0 19 18v-3" /></svg></span>{{ ui.removeFromCollection || 'Remove from collection' }}
        </button>
        <button type="button" class="ctx-item ctx-danger" @click="ctxDeleteDoc">
          <span class="ctx-ic" aria-hidden="true"><svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M4 7h16M7 7l.8 11a2 2 0 0 0 2 1.9h4.4a2 2 0 0 0 2-1.9L17 7M9 7V5.5A1.5 1.5 0 0 1 10.5 4h3A1.5 1.5 0 0 1 15 5.5V7" /></svg></span>{{ ui.deleteDocument }}
        </button>
      </template>
    </div>

    <!-- Floating label following the pointer during an internal drag. Teleported
         to <body>: the sidebar creates a stacking context (z-index 6) that panes
         at 8/20/40/60 paint over, so an in-place ghost vanished the moment it
         crossed into the reader or chat column. -->
    <Teleport to="body">
      <div
        v-if="dragging.active"
        class="drag-ghost"
        :style="{ left: `${dragging.x}px`, top: `${dragging.y}px` }"
      >{{ dragging.label }}</div>
    </Teleport>
    </template>
  </aside>
</template>

<style scoped>
.sidebar {
  position: relative;
  z-index: 6;
  width: 296px;
  min-width: 296px;
  background: var(--surface-3);
  border-right: 1px solid var(--line);
  display: flex;
  flex-direction: column;
  padding: 0 12px 14px 0;
  gap: 14px;
  transition: width var(--dur-slow) var(--ease), min-width var(--dur-slow) var(--ease), padding var(--dur-slow) var(--ease);
}

/* Knowledge-base pivot (P1): icon rail + panel inside the expanded sidebar. */
.sidebar-expanded {
  flex: 1;
  min-height: 0;
  display: flex;
  gap: 4px;
}

.sidebar-rail-strip {
  flex: 0 0 auto;
  width: 32px;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 2px;
  padding: 4px 0 0 4px;
  box-sizing: border-box;
}

.rail-mode {
  width: 28px;
  height: 28px;
  border-radius: var(--r-sm);
  border: 1px solid transparent;
  background: transparent;
  color: var(--ink-2);
  cursor: pointer;
  font-size: 17px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
}

.rail-mode:hover {
  background: var(--surface-hover);
  color: var(--ink);
}

.rail-mode.active {
  background: var(--accent-tint);
  border-color: var(--accent-line);
  color: var(--accent);
}

/* Rail icons are 1em stroke SVGs so they inherit each rail's font-size
   (narrow expanded strip vs larger collapsed buttons). */
.rail-mode svg,
.rail-btn svg {
  display: block;
}

/* Trending is a secondary discovery utility — dimmer at rest so it never competes
   with the core knowledge actions, but fully legible on hover / when active. */
.rail-mode-muted:not(.active),
.rail-btn-muted:not(.active) {
  opacity: 0.5;
}

.rail-mode-muted:hover,
.rail-btn-muted:hover {
  opacity: 1;
}

/* Collapsed rail: the section ribbon fills the width and its icons read a bit
   larger than in the thin expanded strip. */
.collapsed-strip {
  width: 100%;
  gap: 6px;
  padding-top: 4px;
}

.collapsed-strip .rail-mode {
  width: 40px;
  height: 40px;
  font-size: 20px;
  border-radius: var(--r-lg);
}

.rail-divider {
  flex: 0 0 auto;
  width: 26px;
  height: 1px;
  background: var(--surface-hover-strong);
  margin: 8px 0 2px;
}

/* Add menu pops to the right of the narrow rail button (not below it). */
.collapsed-add-menu {
  top: auto;
  bottom: 0;
  right: auto;
  left: calc(100% + 6px);
}

.rail-spacer {
  flex: 1;
}

.sidebar-main {
  flex: 1;
  min-width: 0;
  min-height: 0;
  display: flex;
  flex-direction: column;
  gap: 14px;
  padding-left: 6px;
}

.sidebar.drag-active {
  background: linear-gradient(145deg, var(--accent-tint), transparent 70%);
  border-right-color: var(--accent-line);
}

.sidebar.drag-active::after {
  content: '';
  position: absolute;
  inset: 6px;
  border: 1px dashed var(--accent-line);
  border-radius: var(--r-lg);
  pointer-events: none;
}

.sidebar.collapsed {
  width: 104px;
  min-width: 104px;
  padding: 0 12px 12px;
  align-items: center;
  background:
    linear-gradient(180deg, rgba(255, 255, 255, 0.028), transparent 150px),
    var(--surface-3);
}

.sidebar-window-bar {
  position: relative;
  /* Just enough to clear the macOS traffic lights (≈y20–34); was 56+14≈70px,
     which left a large empty gap above the brand. */
  min-height: 40px;
  width: 100%;
  flex-shrink: 0;
  display: flex;
  align-items: flex-start;
  justify-content: flex-end;
  padding-top: 0;
}

.sidebar.collapsed .sidebar-window-bar {
  min-height: 48px;
  padding-top: 0;
}

.rail-brand {
  width: 58px;
  height: 58px;
  display: grid;
  place-items: center;
  flex: 0 0 auto;
  margin-top: -2px;
  border-radius: var(--r-xl);
  background: var(--surface-wash);
}

.rail-brand img {
  width: 48px;
  height: 48px;
  object-fit: contain;
  filter: drop-shadow(0 2px 4px rgba(0, 0, 0, 0.24));
}

.rail-docs {
  width: 100%;
  min-height: 0;
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
  overflow: auto;
  padding: 6px 0 10px;
}

.rail-docs::-webkit-scrollbar {
  width: 0;
  height: 0;
}

.rail-doc {
  position: relative;
  width: 70px;
  min-height: 74px;
  flex: 0 0 auto;
  display: grid;
  grid-template-rows: 28px auto 8px;
  justify-items: center;
  align-content: center;
  gap: 4px;
  border: 1px solid transparent;
  border-radius: var(--r-lg);
  background: transparent;
  color: var(--ink-2);
  cursor: pointer;
  padding: 8px 7px 7px;
  text-align: center;
  transition: border-color var(--dur-base) var(--ease), background var(--dur-base) var(--ease), color var(--dur-base) var(--ease);
}

.rail-doc:hover {
  background: var(--surface-hover);
  color: var(--ink);
}

.rail-doc.active {
  color: var(--ink);
  border-color: var(--accent-line);
  background: var(--accent-tint);
}

.rail-doc-icon {
  position: relative;
  width: 25px;
  height: 30px;
  border-radius: var(--r-md);
  border: 1px solid var(--line-strong);
  background:
    linear-gradient(135deg, transparent 0 8px, rgba(255, 255, 255, 0.12) 8px 9px, transparent 9px) top right / 12px 12px no-repeat,
    linear-gradient(180deg, rgba(255, 255, 255, 0.11), rgba(255, 255, 255, 0.045));
}

.rail-doc-icon::before {
  content: "";
  position: absolute;
  right: -1px;
  top: -1px;
  width: 10px;
  height: 10px;
  border-left: 1px solid var(--line-strong);
  border-bottom: 1px solid var(--line-strong);
  border-radius: 0 var(--r-sm) 0 var(--r-xs);
  background: var(--surface-hover-strong);
}

.rail-doc-icon::after {
  content: "";
  position: absolute;
  left: 6px;
  right: 6px;
  top: 13px;
  height: 8px;
  border-top: 2px solid rgba(235, 241, 248, 0.58);
  border-bottom: 2px solid rgba(235, 241, 248, 0.42);
}

.rail-doc-name {
  width: 100%;
  min-width: 0;
  color: inherit;
  font-size: 11px;
  line-height: 1.15;
  font-weight: var(--w-medium);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.rail-doc-status {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-height: 8px;
}

.rail-doc-progress {
  position: absolute;
  right: 5px;
  top: 8px;
  bottom: 8px;
  width: 3px;
  overflow: hidden;
  border-radius: var(--r-pill);
  background: var(--surface-hover-strong);
}

.rail-doc-progress span {
  position: absolute;
  left: 0;
  right: 0;
  bottom: 0;
  border-radius: inherit;
  background: linear-gradient(180deg, #8ae8ff, #ffd089);
}

.rail-empty {
  width: 70px;
  min-height: 56px;
  display: grid;
  place-items: center;
  border: 1px dashed var(--line-strong);
  border-radius: var(--r-lg);
  color: var(--ink-3);
  font-size: 11px;
  font-weight: var(--w-medium);
}

.rail-actions {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
  width: 100%;
  flex: 0 0 auto;
  padding-top: 10px;
  border-top: 1px solid var(--line);
}

.rail-btn {
  width: 38px;
  height: 38px;
  border-radius: var(--r-lg);
  border: 1px solid var(--line-strong);
  background:
    linear-gradient(180deg, rgba(255, 255, 255, 0.085), rgba(255, 255, 255, 0.035)),
    rgba(255, 255, 255, 0.028);
  color: var(--ink);
  cursor: pointer;
  display: grid;
  place-items: center;
  font-size: 18px;
  font-weight: var(--w-medium);
  /* The inset highlight is what makes the rail button read as a physical key;
     the raised token supplies the ring and the drop below it. */
  box-shadow:
    inset 0 1px 0 var(--surface-hover-strong),
    var(--shadow-raised);
  transition: border-color var(--dur-base) var(--ease), background var(--dur-base) var(--ease), color var(--dur-base) var(--ease), transform var(--dur-base) var(--ease);
}

.rail-btn:hover:not(:disabled) {
  color: var(--ink);
  border-color: var(--accent-line);
  background:
    linear-gradient(180deg, rgba(122, 162, 255, 0.18), rgba(122, 162, 255, 0.065)),
    rgba(255, 255, 255, 0.045);
  transform: translateY(-1px);
}

.rail-btn:disabled {
  opacity: 0.42;
  cursor: not-allowed;
}

.sidebar-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  min-width: 0;
}

.panel-label {
  font-size: 12px;
  font-weight: var(--w-strong);
  letter-spacing: var(--tracking-caps);
  text-transform: uppercase;
  color: var(--ink-3);
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.panel-actions {
  display: flex;
  gap: 2px;
  flex: 0 0 auto;
}

.panel-action-btn {
  width: 24px;
  height: 24px;
  display: grid;
  place-items: center;
  border: none;
  border-radius: var(--r-sm);
  background: transparent;
  color: var(--ink-2);
  font-size: 15px;
  line-height: 1;
  cursor: pointer;
  transition: background var(--dur-fast) var(--ease), color var(--dur-fast) var(--ease);
}

.panel-action-btn:hover:not(:disabled) {
  background: var(--surface-hover);
  color: var(--ink);
}

.panel-action-btn:disabled {
  opacity: 0.4;
  cursor: default;
}

.add-menu-wrap {
  position: relative;
}

.add-menu {
  position: absolute;
  top: calc(100% + 4px);
  right: 0;
  z-index: 20;
  min-width: 168px;
  padding: 4px;
  border: 1px solid var(--line);
  border-radius: var(--r-md);
  background: var(--surface-1);
  box-shadow: var(--shadow-overlay);
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.add-menu-item {
  display: flex;
  align-items: center;
  gap: 8px;
  width: 100%;
  padding: 7px 9px;
  border: none;
  border-radius: var(--r-sm);
  background: transparent;
  color: var(--ink);
  font-size: 13px;
  text-align: left;
  cursor: pointer;
}

.add-menu-item:hover {
  background: var(--surface-hover);
}

.add-menu-ic {
  display: grid;
  place-items: center;
  width: 16px;
  height: 16px;
  flex: 0 0 16px;
  color: var(--ink-2);
}

.add-menu-ic svg {
  display: block;
}

/* Right-click context menu (fixed to the cursor). */
.ctx-menu {
  position: fixed;
  z-index: 1000;
  min-width: 172px;
  padding: 4px;
  border: 1px solid var(--line);
  border-radius: var(--r-md);
  background: var(--surface-1);
  box-shadow: var(--shadow-overlay);
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.ctx-item {
  display: flex;
  align-items: center;
  gap: 9px;
  width: 100%;
  padding: 7px 10px;
  border: none;
  border-radius: var(--r-sm);
  background: transparent;
  color: var(--ink);
  font-size: 13px;
  text-align: left;
  cursor: pointer;
}

.ctx-item:hover {
  background: var(--surface-hover);
}

.ctx-item.ctx-danger:hover {
  background: var(--danger-tint);
  color: var(--danger);
}

/* The icon is dimmer than its label at rest; on the danger row's hover it should
   follow the label into red rather than stay grey. */
.ctx-item.ctx-danger:hover .ctx-ic {
  color: inherit;
}

.ctx-ic {
  display: grid;
  place-items: center;
  width: 16px;
  height: 16px;
  flex: 0 0 16px;
  color: var(--ink-2);
}

.ctx-ic svg {
  display: block;
}

.ctx-sep {
  height: 1px;
  margin: 3px 4px;
  background: var(--line);
}

.panel-hint {
  margin-left: auto;
  font-size: 11px;
  color: var(--ink-3);
  opacity: 0.7;
}

.path-reveal-btn {
  width: 22px;
  height: 22px;
  flex: 0 0 22px;
  border-radius: var(--r-sm);
  border: 1px solid transparent;
  background: transparent;
  color: var(--ink-3);
  cursor: pointer;
  display: grid;
  place-items: center;
  padding: 0;
  font-size: 12px;
  line-height: 1;
  opacity: 0;
  transition: opacity var(--dur-base) var(--ease), background var(--dur-base) var(--ease), color var(--dur-base) var(--ease), border-color var(--dur-base) var(--ease);
}

.sidebar-header:hover .path-reveal-btn,
.path-reveal-btn:focus-visible {
  opacity: 1;
}

.path-reveal-btn:hover:not(:disabled) {
  color: var(--ink);
  border-color: var(--line);
  background: var(--surface-hover);
}

.path-reveal-btn:disabled {
  cursor: not-allowed;
  opacity: 0;
}


.search-box {
  display: flex;
  align-items: center;
  gap: 8px;
  border: 1px solid var(--line);
  background: var(--surface-wash);
  border-radius: var(--r-lg);
  padding: 0 12px;
  min-height: 40px;
}

.search-box input {
  width: 100%;
  background: transparent;
  border: none;
  color: var(--ink);
  outline: none;
  font-size: 13px;
}

.search-icon {
  flex-shrink: 0;
  display: inline-flex;
  align-items: center;
  color: var(--ink-3);
}

.tree-area {
  flex: 1;
  min-width: 0;
  min-height: 0;
  overflow-x: hidden;
  overflow-y: auto;
  padding-right: 4px;
}

/* "Trending Papers" discovery entry, pinned above the local folders. */
/* Scrollbar look comes from the global style in styles/main.css. */

.folder-group + .folder-group {
  margin-top: 0px;
}

.folder-group {
  position: relative;
  border-radius: var(--r-lg);
  padding: 0px 4px 0px;
  margin-top: -6px;
  transition: background var(--dur-base) var(--ease), box-shadow var(--dur-base) var(--ease);
}

.folder-group.drop-target {
  background: var(--accent-tint);
  box-shadow:
    inset 0 0 0 1px var(--accent-line),
    0 0 0 1px var(--accent-tint);
}

.folder-group.drop-target .folder-title {
  color: var(--ink);
}

.workspace-title-row {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-bottom: 8px;
}

.folder-title {
  flex: 1;
  min-width: 0;
  display: flex;
  align-items: center;
  gap: 6px;
  border: none;
  background: transparent;
  color: var(--ink-2);
  cursor: pointer;
  padding: 4px 2px;
  text-align: left;
  font-size: 12px;
  font-weight: var(--w-strong);
}

.folder-title:hover {
  color: var(--ink);
}

.folder-caret {
  width: 12px;
  flex: 0 0 12px;
  color: var(--ink-3);
}

.folder-name {
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.folder-open-btn {
  width: 24px;
  height: 24px;
  flex: 0 0 24px;
  display: grid;
  place-items: center;
  border: 1px solid transparent;
  border-radius: var(--r-sm);
  background: transparent;
  color: var(--ink-3);
  cursor: pointer;
  opacity: 0;
  transition: opacity var(--dur-base) var(--ease), background var(--dur-base) var(--ease), color var(--dur-base) var(--ease), border-color var(--dur-base) var(--ease);
  padding: 0;
  line-height: 1;
  font-size: 14px;
}

.folder-action-icon {
  display: block;
  line-height: 1;
  pointer-events: none;
}

.workspace-title-row:hover .folder-open-btn,
.folder-open-btn:focus-visible {
  opacity: 1;
}

.folder-open-btn:hover {
  color: var(--ink);
  border-color: var(--line);
  background: var(--surface-hover);
}

.folder-delete-btn {
  color: var(--danger);
}

.folder-delete-icon {
  position: relative;
  left: 0;
}

.folder-delete-btn:hover {
  color: var(--danger);
  border-color: var(--danger-line);
  background: var(--danger-tint);
}

/* Knowledge-base pivot: collection tree rows. */
.collection-tree {
  display: flex;
  flex-direction: column;
  gap: 2px;
  /* Tree rows are controls, not prose: never let a press start a text selection
     (belt-and-braces with preventDefault on mousedown, which also covers the
     press landing on a child span). */
  user-select: none;
  -webkit-user-select: none;
}

.collection-row {
  position: relative;
  display: flex;
  align-items: center;
  gap: 4px;
  min-width: 0;
  border-radius: var(--r-md);
  padding: 4px 6px 4px 2px;
  border: 1px solid transparent;
  color: var(--ink-2);
}

/* Manual reorder: an accent insertion line at the row edge where a dragged
   sibling would land. doc-row is already position:relative. */
.collection-row.drop-before::before,
.collection-row.drop-after::after,
.doc-row.drop-before::before,
.doc-row.drop-after::after {
  content: '';
  position: absolute;
  left: 4px;
  right: 4px;
  height: 2px;
  border-radius: var(--r-xs);
  background: var(--accent);
  pointer-events: none;
}

/* Sit the line on the row's inner edge, not in the inter-row gap: .doc-row clips
   to its box (overflow:hidden for the WKWebView ellipsis fix), so a negative
   offset would vanish on documents. */
.collection-row.drop-before::before,
.doc-row.drop-before::before {
  top: 0;
}

.collection-row.drop-after::after,
.doc-row.drop-after::after {
  bottom: 0;
}

/* Floating label that follows the pointer during a mouse-based drag. */
.drag-ghost {
  position: fixed;
  z-index: 2000;
  transform: translate(12px, 8px);
  max-width: 220px;
  padding: 4px 10px;
  border-radius: var(--r-md);
  background: var(--accent);
  color: var(--on-fill);
  font-size: 12px;
  font-weight: var(--w-medium);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  pointer-events: none;
  box-shadow: var(--shadow-overlay);
}

.collection-row:hover {
  background: var(--surface-wash);
}

/* C-d: highlight a collection as a drop target (doc move / nest / OS file). */
.collection-row.drag-over {
  background: var(--accent-tint);
  border-color: var(--accent-line);
  color: var(--ink);
}

.collection-caret {
  width: 16px;
  height: 16px;
  flex: 0 0 16px;
  display: grid;
  place-items: center;
  border: none;
  background: transparent;
  color: var(--ink-3);
  cursor: pointer;
  padding: 0;
  line-height: 1;
}

.collection-caret svg {
  transition: transform var(--dur-fast) var(--ease);
}

.collection-caret.expanded svg {
  transform: rotate(90deg);
}

.collection-row:hover .collection-caret {
  color: var(--ink-2);
}

.collection-name-btn {
  flex: 1;
  min-width: 0;
  display: flex;
  align-items: center;
  border: none;
  background: transparent;
  color: inherit;
  cursor: pointer;
  padding: 2px 0;
  text-align: left;
}

  /* Folder rows read as headings (600), documents as leaves (below) — same size,
     so the tree has one type scale instead of children larger than their parent. */
.collection-name {
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-size: var(--fs-body);
  font-weight: var(--w-strong);
}

.collection-count {
  flex: 0 0 auto;
  color: var(--ink-3);
  font-size: 11px;
  font-variant-numeric: tabular-nums;
}

.collection-actions {
  display: flex;
  gap: 2px;
  flex: 0 0 auto;
  opacity: 0;
  transition: opacity var(--dur-fast) var(--ease);
}

.collection-row:hover .collection-actions,
.collection-actions:focus-within,
.collection-actions.collection-confirm {
  opacity: 1;
}

/* Inline delete confirm: keep the ✓/✗ visible regardless of hover. */
.collection-confirm-yes {
  color: var(--danger);
}

.collection-action-btn {
  width: 20px;
  height: 20px;
  display: grid;
  place-items: center;
  border: none;
  border-radius: var(--r-sm);
  background: transparent;
  color: var(--ink-3);
  line-height: 1;
  cursor: pointer;
  transition: background var(--dur-fast) var(--ease), color var(--dur-fast) var(--ease);
}

.collection-action-btn svg {
  display: block;
}

.collection-action-btn:hover {
  background: var(--surface-hover-strong);
  color: var(--ink);
}

.collection-delete-btn {
  color: var(--danger);
}

.collection-delete-btn:hover {
  background: var(--danger-tint);
  color: var(--danger);
}

.collection-rename-input {
  flex: 1;
  min-width: 0;
  border: 1px solid var(--line);
  border-radius: var(--r-sm);
  background: var(--surface-hover);
  color: var(--ink);
  font-size: 12px;
  font-weight: var(--w-medium);
  padding: 2px 6px;
  outline: none;
}

.doc-row {
  width: 100%;
  background: transparent;
  border: none;
  color: inherit;
  text-align: left;
  cursor: pointer;
}

.doc-row {
  position: relative;
  display: flex;
  flex-direction: column;
  gap: 5px;
  /* A document is a leaf of the collection tree, not a floating card: match the
     collection-row rhythm (tight padding, no inter-row margin, small radius) so
     parent folders and their documents share one vertical cadence.
     Left padding = collection-row's 2px + 16px caret + 4px gap, so root-level
     files line up with folder titles instead of hanging out past the caret. */
  padding: 4px 8px 4px 22px;
  border-radius: var(--r-sm);
  margin-bottom: 0;
  border: 1px solid transparent;
  /* WKWebView (Tauri/Safari) sizes <button> to its min-content width and won't
     honor width:100% shrink, breaking the ellipsis chain. Force shrink + clip. */
  min-width: 0;
  max-width: 100%;
  overflow: hidden;
}

/* Per-document actions (reindex + delete), revealed on row hover. */
.doc-row:hover {
  background: var(--surface-wash);
}

.doc-row.active {
  background: var(--surface-hover-strong);
  border-color: var(--accent-line);
}

.doc-main {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 8px;
}

.doc-name-wrap {
  display: flex;
  align-items: baseline;
  gap: 7px;
  flex: 1;
  min-width: 0;
}

.doc-status-dot {
  flex-shrink: 0;
  align-self: center;
  width: 6px;
  height: 6px;
  border-radius: var(--r-pill);
  background: var(--ink-3);
}

.doc-status-dot.failed {
  background: var(--danger);
}

.doc-name {
  flex: 1;
  min-width: 0;
  color: var(--ink);
  font-size: var(--fs-body);
  line-height: 1.35;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.doc-time {
  flex-shrink: 0;
  color: var(--ink-3);
  font-size: var(--fs-caption);
  white-space: nowrap;
}

.doc-progress {
  height: 3px;
  overflow: hidden;
  border-radius: var(--r-pill);
  background: var(--surface-hover-strong);
}

.doc-progress span {
  display: block;
  height: 100%;
  border-radius: inherit;
  background: linear-gradient(90deg, #ffd089, #8ae8ff);
  transition: width var(--dur-slow) var(--ease);
}

.empty-tree,
.scan-error {
  border: 1px solid var(--line);
  border-radius: var(--r-lg);
  padding: 12px;
  color: var(--ink-2);
  font-size: 13px;
  line-height: 1.5;
}

.scan-error {
  color: var(--danger);
  border-color: var(--danger-line);
  background: var(--danger-tint);
}

.sidebar-status {
  flex: 0 0 auto;
  padding: 6px 10px 2px;
  font-size: 11px;
  color: var(--ink-3);
  min-height: 16px;
}
</style>
