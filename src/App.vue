<script setup>
import {
  computed,
  defineAsyncComponent,
  h,
  nextTick,
  onBeforeUnmount,
  onMounted,
  reactive,
  ref,
  watch,
} from 'vue'
import { invoke } from '@tauri-apps/api/core'
import { listen } from '@tauri-apps/api/event'
import { translationLanguages } from './mockData'
import { messages } from './i18n'
import { normalizeLinkedBlockHover } from './translationLinking'
import { usePersistedRef, readPersisted, writePersisted } from './persistedState'
import { themePreference, setThemePreference } from './theme'

const UNCONFIGURED_CHAT_MODEL_ID = 'unconfigured-model'
const ASSISTANT_STREAM_DRAIN_MS = 35
const ASSISTANT_STREAM_CHARS_PER_TICK = 2
// When the visible text falls far behind the target — which happens when a
// non-streaming path (the tool-capable agent loop, or a local CLI) delivers the
// whole answer in one shot — reveal a fraction of the backlog each tick instead
// of a fixed two chars. Without this a 4k-char answer takes ~70s to paint, which
// leaves the message stuck in the "running" state (stop button visible, nothing
// to cancel) long after the backend finished. /6 drains a one-shot answer in
// ~1s while a genuine token stream stays smooth.
const ASSISTANT_STREAM_CATCHUP_DIVISOR = 6
const CHAT_STREAM_DEBUG_STORAGE_KEY = 'lumenfolio.chatStreamDebug'
const ASYNC_COMPONENT_TIMEOUT_MS = 30000
const CLEAR_CHAT_HISTORY_TIMEOUT_MS = 5000
const WORKSPACE_FILE_DROP_DEBOUNCE_MS = 900
const AsyncPanelLoading = {
  render: () => h('div', {
    class: 'async-panel-loading',
    'aria-hidden': 'true',
  }),
}
const WorkspaceSidebar = defineAsyncComponent({
  loader: () => import('./components/WorkspaceSidebar.vue'),
  loadingComponent: AsyncPanelLoading,
  delay: 80,
  timeout: ASYNC_COMPONENT_TIMEOUT_MS,
})
// Async with no loading component: the palette opens on a keystroke, and a
// spinner flashing in the middle of the screen would be worse than the ~1 frame
// the chunk takes to arrive.
const CommandPalette = defineAsyncComponent({
  loader: () => import('./components/CommandPalette.vue'),
  timeout: ASYNC_COMPONENT_TIMEOUT_MS,
})
const ReaderPane = defineAsyncComponent({
  loader: () => import('./components/ReaderPane.vue'),
  loadingComponent: AsyncPanelLoading,
  delay: 80,
  timeout: ASYNC_COMPONENT_TIMEOUT_MS,
})
const TrendingFeed = defineAsyncComponent({
  loader: () => import('./components/TrendingFeed.vue'),
  loadingComponent: AsyncPanelLoading,
  delay: 80,
  timeout: ASYNC_COMPONENT_TIMEOUT_MS,
})
const KnowledgePane = defineAsyncComponent({
  loader: () => import('./components/KnowledgePane.vue'),
  delay: 120,
  timeout: ASYNC_COMPONENT_TIMEOUT_MS,
})
const NoteEditor = defineAsyncComponent({
  loader: () => import('./components/NoteEditor.vue'),
  loadingComponent: AsyncPanelLoading,
  delay: 80,
  timeout: ASYNC_COMPONENT_TIMEOUT_MS,
})
const OfficeViewer = defineAsyncComponent({
  loader: () => import('./components/OfficeViewer.vue'),
  loadingComponent: AsyncPanelLoading,
  delay: 80,
  timeout: ASYNC_COMPONENT_TIMEOUT_MS,
})
const KnowledgeGraphView = defineAsyncComponent({
  loader: () => import('./components/KnowledgeGraphView.vue'),
  loadingComponent: AsyncPanelLoading,
  delay: 80,
  timeout: ASYNC_COMPONENT_TIMEOUT_MS,
})
const ChatPane = defineAsyncComponent({
  loader: () => import('./components/ChatPane.vue'),
  loadingComponent: AsyncPanelLoading,
  delay: 80,
  timeout: ASYNC_COMPONENT_TIMEOUT_MS,
})
const NotesPane = defineAsyncComponent({
  loader: () => import('./components/NotesPane.vue'),
  loadingComponent: AsyncPanelLoading,
  delay: 80,
  timeout: ASYNC_COMPONENT_TIMEOUT_MS,
})
const NoteComposer = defineAsyncComponent({
  loader: () => import('./components/NoteComposer.vue'),
})
const MODEL_PROVIDER_PRESETS = {
  'openai-compatible': {
    name: 'OpenAI Compatible',
    baseUrl: 'https://api.openai.com/v1',
    model: '',
  },
  openai: {
    name: 'OpenAI',
    baseUrl: 'https://api.openai.com/v1',
    model: 'gpt-4.1-mini',
  },
  deepseek: {
    name: 'DeepSeek',
    baseUrl: 'https://api.deepseek.com',
    model: 'deepseek-v4-flash',
  },
  openrouter: {
    name: 'OpenRouter',
    baseUrl: 'https://openrouter.ai/api/v1',
    model: 'openai/gpt-4.1-mini',
  },
}
const MODEL_CAPABILITY_OPTIONS = ['vision', 'reasoning', 'tool_use']

const workspace = reactive({
  roots: [],
})
// Knowledge-base pivot: user-authored logical collections (nestable folders
// decoupled from disk). Flat list of { id, parentId, name, position }.
const collections = ref([])
// The collection new imports/notes/clips file into; also the highlighted tree row.
const selectedCollectionId = usePersistedRef('selectedCollectionId', null)
const locale = usePersistedRef('locale', 'en')
const ui = computed(() => messages[locale.value] || messages.en)
const filter = ref('')
const selectedDocId = ref('')
// IDE-style document tabs: an ordered working set of opened documents, layered on
// top of selectedDocId (the active tab). Restored (filtered to still-existing docs)
// in loadLastWorkspace.
const openTabs = ref([])
// Carries a citation across a cross-document jump: openTab() changes selectedDocId,
// which triggers watch(selectedDocument) that resets activePage to the doc's saved
// page. We stash the jump target here and re-apply it after that reset (nextTick).
let pendingCitationJump = null
// Remember the active document across restarts; restored (with existence
// validation) in loadLastWorkspace. Skip empty transient values.
watch(selectedDocId, (id) => {
  if (id) writePersisted('selectedDocId', id)
})
// openTabs is only ever reassigned (never mutated in place), so a shallow watch
// fires on every change — no deep traversal needed.
watch(openTabs, (tabs) => {
  writePersisted('openTabs', tabs)
})
const translationLang = ref('zh')
const viewMode = ref('original')
const activePage = ref(1)
const activeBlockId = ref('')
const activeCitationId = ref('')
const activeHighlight = ref(null)
// The citation last clicked, for viewers that anchor on text rather than on PDF
// geometry (the Office preview).
const activeCitation = ref(null)
const hoveredLinkedBlock = ref(null)
const lastSelection = ref(null)
const activeTranslation = ref(null)
const inlineTranslateOpen = ref(false)
const leftCollapsed = usePersistedRef('leftCollapsed', false)
const rightCollapsed = usePersistedRef('rightCollapsed', false)
// Per-root folder collapse state, keyed by root id (path as fallback). Survives
// restarts so a folder the user collapsed stays collapsed next launch.
const collapsedRoots = usePersistedRef('collapsedRoots', {})
const rightWidth = usePersistedRef('rightWidth', 500, { debounceMs: 300 })
// Notes is a floating drawer that slides in over the Agent pane (not a mutually
// exclusive tab), so the conversation stays visible underneath. Default closed.
const notesDrawerOpen = ref(false)
const activeNoteId = ref('')
// Trending Papers (online discovery) state.
// Opt-out for the local-first audience: when off, nothing is fetched and the
// discovery entry/feed are hidden.
const trendingEnabled = usePersistedRef('trendingEnabled', true)
// Notes vault: the folder notes are mirrored to as plain .md. Lives in the
// backend settings table (not localStorage) because the Rust side writes the
// files. Empty string = mirroring off.
const vaultDir = ref('')
const vaultBusy = ref(false)
const vaultStatus = ref('')

// Knowledge API: a resident local MCP endpoint external harnesses can query.
// Off by default — while it runs, any process on this machine holding the token
// can read the whole library without going through the app.
const knowledgeApi = ref({
  enabled: false,
  port: 37650,
  token: '',
  url: '',
  running: false,
  error: '',
})
const knowledgeApiBusy = ref(false)
const knowledgeApiTokenVisible = ref(false)

// Masked by default so the token does not sit in plain sight during a screen
// share; the copy buttons work regardless of whether it is revealed.
const knowledgeApiTokenDisplay = computed(() => {
  const token = knowledgeApi.value.token || ''
  if (!token) return ''
  return knowledgeApiTokenVisible.value ? token : `${token.slice(0, 4)}${'•'.repeat(12)}`
})

const knowledgeApiUrl = computed(() => (
  knowledgeApi.value.url || `http://127.0.0.1:${knowledgeApi.value.port}/mcp`
))

// Ready-to-paste registration for the two harnesses most likely to be used.
const knowledgeApiClaudeCommand = computed(() => (
  `claude mcp add --transport http lumenfolio ${knowledgeApiUrl.value}`
  + ` --header "Authorization: Bearer ${knowledgeApi.value.token}"`
))
const knowledgeApiCodexCommand = computed(() => (
  `codex mcp add lumenfolio --url ${knowledgeApiUrl.value}`
  + ` --bearer-token "${knowledgeApi.value.token}"`
))

async function loadKnowledgeApi() {
  try {
    knowledgeApi.value = await invoke('load_knowledge_api_settings')
  } catch (err) {
    console.warn('load_knowledge_api_settings failed', err)
  }
}

async function saveKnowledgeApi() {
  knowledgeApiBusy.value = true
  try {
    knowledgeApi.value = await invoke('save_knowledge_api_settings', {
      enabled: knowledgeApi.value.enabled,
      port: Number(knowledgeApi.value.port) || 37650,
    })
  } catch (err) {
    knowledgeApi.value.error = err?.message || String(err)
  } finally {
    knowledgeApiBusy.value = false
  }
}

async function rotateKnowledgeApiToken() {
  knowledgeApiBusy.value = true
  try {
    knowledgeApi.value = await invoke('rotate_knowledge_api_token')
    showToast(ui.value.knowledgeApiRotated || 'New token issued')
  } catch (err) {
    knowledgeApi.value.error = err?.message || String(err)
  } finally {
    knowledgeApiBusy.value = false
  }
}

async function copyText(text, message) {
  try {
    await navigator.clipboard?.writeText(text)
    showToast(message)
  } catch (err) {
    console.warn('clipboard write failed', err)
  }
}

// Database snapshots. Notes are already safe as .md; this covers what is left —
// collections, chat history, settings — none of which can be regenerated.
const backupDir = ref('')
const backupKeep = ref(10)
// Hours between automatic snapshots; 0 = manual only (the default).
const backupIntervalHours = ref(0)
const backupEntries = ref([])
const backupBusy = ref(false)
const backupStatus = ref('')

function applyBackupSettings(settings) {
  backupDir.value = settings?.dir || ''
  backupKeep.value = settings?.keep || 10
  backupIntervalHours.value = settings?.intervalHours || 0
  backupEntries.value = settings?.entries || []
}

async function loadBackupSettings() {
  try {
    applyBackupSettings(await invoke('load_backup_settings'))
  } catch (err) {
    console.warn('load_backup_settings failed', err)
  }
}

async function saveBackupSettings() {
  try {
    await invoke('save_backup_settings', {
      dir: backupDir.value || '',
      keep: Number(backupKeep.value) || 10,
      intervalHours: Number(backupIntervalHours.value) || 0,
    })
    await loadBackupSettings()
  } catch (err) {
    backupStatus.value = err?.message || String(err)
  }
}

async function chooseBackupDir() {
  try {
    const picked = await invoke('choose_backup_dir')
    if (!picked) return
    backupDir.value = picked
    await saveBackupSettings()
  } catch (err) {
    backupStatus.value = err?.message || String(err)
  }
}

async function backupNow() {
  backupBusy.value = true
  backupStatus.value = ''
  try {
    applyBackupSettings(await invoke('create_backup_now'))
    backupStatus.value = ui.value.backupDone || 'Backed up'
  } catch (err) {
    backupStatus.value = err?.message || String(err)
  } finally {
    backupBusy.value = false
  }
}

async function restoreBackup(entry) {
  if (!entry?.path) return
  backupBusy.value = true
  backupStatus.value = ''
  try {
    // Staged, not applied: swapping a database the app has open is the very
    // corruption this feature exists to avoid, so it lands on next launch.
    await invoke('restore_backup', { path: entry.path })
    backupStatus.value = ui.value.backupRestoreStaged || 'Restart to finish restoring'
  } catch (err) {
    backupStatus.value = err?.message || String(err)
  } finally {
    backupBusy.value = false
  }
}

async function loadVaultSettings() {
  try {
    const settings = await invoke('load_vault_settings')
    vaultDir.value = settings?.dir || ''
  } catch (err) {
    console.warn('load_vault_settings failed', err)
  }
}

async function saveVaultDir() {
  vaultBusy.value = true
  vaultStatus.value = ''
  try {
    const written = await invoke('set_vault_dir', { dir: vaultDir.value || '' })
    // Changing the folder re-exports everything, so the new location is complete
    // immediately rather than only holding notes saved from now on.
    vaultStatus.value = vaultDir.value
      ? `${ui.value.notesVaultExported || 'Exported'} ${written}`
      : ''
  } catch (err) {
    vaultStatus.value = err?.message || String(err)
  } finally {
    vaultBusy.value = false
  }
}

async function chooseVaultDir() {
  try {
    const picked = await invoke('choose_vault_dir')
    if (!picked) return
    vaultDir.value = picked
    await saveVaultDir()
  } catch (err) {
    vaultStatus.value = err?.message || String(err)
  }
}

async function exportVault() {
  vaultBusy.value = true
  try {
    const written = await invoke('export_notes_to_vault')
    vaultStatus.value = `${ui.value.notesVaultExported || 'Exported'} ${written}`
  } catch (err) {
    vaultStatus.value = err?.message || String(err)
  } finally {
    vaultBusy.value = false
  }
}

async function importVault() {
  vaultBusy.value = true
  try {
    const result = await invoke('import_notes_from_vault')
    vaultStatus.value = `${ui.value.notesVaultImported || 'Recovered'} ${result?.imported ?? 0}`
    if (result?.snapshot) upsertWorkspaceRootSnapshot(result.snapshot)
  } catch (err) {
    vaultStatus.value = err?.message || String(err)
  } finally {
    vaultBusy.value = false
  }
}
// Knowledge precipitation (Stream 1 LLM extraction). Default on; when off, no
// per-document LLM extraction is triggered. Turning it on triggers a backfill.
const knowledgeEnabled = usePersistedRef('knowledgeEnabled', true)
// Chat web search ("联网"). Default off; when on, the agent is offered the
// web_search / web_fetch tools (Exa when a key is configured, else DuckDuckGo).
const webSearchEnabled = usePersistedRef('webSearchEnabled', false)
// Knowledge card for the currently-open document (loaded from get_document_knowledge).
const knowledgeCard = ref(null) // { status, summary, entities[], concepts[], keywords[], error }
// Reader-area 阅读/知识 tab: true shows the full-area knowledge view over the
// PDF. Sticky across document switches so graph-hopping (click a related paper
// → land on ITS knowledge page) works as a navigation flow.
const knowledgeView = ref(false)
// Related papers for the open document (P3a): [{ documentId, title, score, sharedConcepts[], coCitation }].
const relatedDocs = ref([])
// Inter-links among those related docs (mini ego-graph cluster structure).
const relatedLinks = ref([])
// Knowledge graph view (P3b): full-screen force graph in the reader area.
const graphView = ref(false)
// When opened from the knowledge card's "view in graph" button, start the graph
// focused (ego view) on the current document instead of the whole library.
const graphFocusOnOpen = ref(false)
const graphData = ref(null)
const graphStatus = ref('idle') // idle | loading | loaded | failed
const graphError = ref('')
let knowledgeConsolidated = false // run alias consolidation at most once per session
const showGraph = computed(() => graphView.value && !showTrending.value)
// When browsing a non-reader surface, the chat focus row shows this instead of a
// stale focus-document title (the agent still has the trending/graph tools).
const chatBrowsingLabel = computed(() => {
  if (showTrending.value) {
    const periodLabel = ui.value[`trending${trendingPeriod.value.charAt(0).toUpperCase()}${trendingPeriod.value.slice(1)}`] || ''
    return `${periodLabel} ${ui.value.trendingPapers}`.trim()
  }
  if (showGraph.value) return ui.value.knowledgeGraph
  // KB pivot (P1): no focus document → the chat is library-wide ("ask my
  // knowledge base"), so label it as such instead of leaving it blank.
  const focus = activeFocusDoc.value
  if (!focus || focus.id === 'empty') return ui.value.askKnowledgeBase
  return ''
})
// Persisted so the discovery feed re-opens on next launch if that's where the
// user left off. Note: restoring the last document flips this off mid-load, so
// the value is re-applied after the workspace settles (loadLastWorkspaceAfterFirstPaint).
const trendingView = usePersistedRef('trendingView', false)
// daily | weekly | monthly — which HF papers feed the Trending view shows.
const trendingPeriod = usePersistedRef('trendingPeriod', 'daily')
const trendingPapers = ref([])
const trendingStatus = ref('idle') // idle | loading | loaded | failed
const trendingError = ref('')
const trendingAddError = ref('')
const trendingAddingIds = ref([])
// Update notification (notify-only): checked once after startup. The banner is
// suppressed once the user dismisses a given version (persisted), so it won't nag.
const updateInfo = ref(null) // { current, latest, hasUpdate, url }
const dismissedUpdateVersion = usePersistedRef('dismissedUpdateVersion', '')
const showUpdateBanner = computed(() => (
  Boolean(updateInfo.value?.hasUpdate)
  && updateInfo.value.latest !== dismissedUpdateVersion.value
))
// Note composer state for create/edit.
const noteComposer = ref({ open: false, mode: 'create', quoteText: '', content: '', noteId: '', selection: null })
const noteComposerSaving = ref(false)
const noteDeleteConfirmOpen = ref(false)
const noteDeleteStatus = ref('idle')
const noteDeleteError = ref('')
const noteDeleteTarget = ref({
  id: '',
  documentId: '',
  page: 0,
  quoteText: '',
  content: '',
})
const chatFocusRequest = ref(0)
const viewerReloadKey = ref(0)
const selectedChatModelId = ref(UNCONFIGURED_CHAT_MODEL_ID)
// Last chat model the user explicitly picked, persisted so reopening the app
// restores it instead of falling back to the default provider's first model.
const lastChatModelId = usePersistedRef('lastChatModelId', '')
const translationProvider = ref('google-web')
let syncingTranslationLangFromDocument = false
const translationFallbackEnabled = ref(true)
const workspaceStatus = ref('idle')
const workspaceError = ref('')
const modelProviders = ref([])
const editableProviders = ref([])
const selectedProviderEditKey = ref('')
const settingsOpen = ref(false)
const paletteOpen = ref(false)
// Order is deliberate: the default sits last so the row reads
// least-committal → most-committal rather than putting "system" in the middle.
const themeChoices = computed(() => [
  { id: 'system', label: ui.value.appearanceSystem },
  { id: 'light', label: ui.value.appearanceLight },
  { id: 'dark', label: ui.value.appearanceDark },
])
const settingsSection = usePersistedRef('settingsSection', 'chat')
// Locally-installed agent CLIs (Codex / Claude Code) offered as zero-config chat
// providers. P0: detection + status only. [{ kind, label, installed, version, path, installUrl }]
const localAgentStatus = ref([])
const localAgentChecking = ref(false)
// Per-kind "Test connection" state: { [kind]: { testing, ok, message } }.
const localAgentTest = ref({})
// Hard-coded default order is Codex → Claude, but the preferred one is configurable.
const preferredLocalAgent = usePersistedRef('preferredLocalAgent', 'codex')
const settingsStatus = ref('idle')
const settingsError = ref('')
const clearChatConfirmOpen = ref(false)
const clearChatStatus = ref('idle')
const clearChatError = ref('')
const clearChatTargetSessionId = ref('')
const clearChatTargetTitle = ref('')
const removeWorkspaceRootConfirmOpen = ref(false)
const removeWorkspaceRootStatus = ref('idle')
const removeWorkspaceRootError = ref('')
const removeWorkspaceRootTarget = ref({
  id: '',
  name: '',
  path: '',
  docCount: 0,
})
const documentDeleteConfirmOpen = ref(false)
const documentDeleteStatus = ref('idle')
const documentDeleteError = ref('')
const documentDeleteTarget = ref({ id: '', title: '', isNote: false })
// Provider delete confirm — window.confirm() no-ops in Tauri's WKWebView, so
// destructive provider removal goes through an in-app modal like the others.
const providerDeleteConfirmOpen = ref(false)
const providerDeleteStatus = ref('idle')
const providerDeleteError = ref('')
const providerDeleteTarget = ref({ key: '', id: '', name: '' })
// The mounted NoteEditor (only present while an editable source is open). Used to
// flush pending edits to the database before the agent reads the note.
const noteEditorRef = ref(null)
const workspaceDropActive = ref(false)
const workspaceDropTargetRootId = ref('')
// C-d: collection row (or '__unfiled__') an OS-file drag is hovering over.
const workspaceDropTargetCollectionId = ref('')
const lastWorkspaceDropAt = ref(0)
const ignoreNextTauriFileDrop = ref(false)
const providerTestStatus = ref('idle')
const providerTestMessage = ref('')
const modelFetchStatus = ref('idle')
const modelFetchMessage = ref('')
let localProviderDraftCounter = 1
let localModelDraftCounter = 0
const providerForm = reactive(createEmptyProviderForm())
const microsoftForm = reactive(createEmptyMicrosoftForm())
// Exa web-search key. `apiKey` is the editable field (blank = unchanged on save);
// `hasApiKey` reflects whether a key is stored, `status` drives the save button.
const webSearchForm = reactive({ apiKey: '', hasApiKey: false, status: 'idle' })
// App-wide outbound proxy (e.g. to reach huggingface.co behind a firewall).
const proxyForm = reactive({ proxyUrl: '', status: 'idle' })
const pdfTranslationRuntime = ref({
  checked: false,
  ok: false,
  error: '',
})

let dragCleanup = null
let translationTimer = null
let translationPageRefreshTimer = null
const translationSinglePageRefreshTimers = new Map()
const translationPageLoadsInFlight = new Set()
let providerTypeSyncSuspended = false
let agentActivityUnlisten = null
let answerDeltaUnlisten = null
let reasoningDeltaUnlisten = null
let askDocumentDoneUnlisten = null
let askDocumentErrorUnlisten = null
let askDocumentStoppedUnlisten = null
let documentIndexUnlisten = null
let visualIndexUnlisten = null
let knowledgeUnlisten = null
let translationJobUnlisten = null
let pdfTranslationUnlisten = null
let localMessageCounter = 0
let dragEnterUnlisten = null
let dragOverUnlisten = null
let dragLeaveUnlisten = null
let dragDropUnlisten = null
let fileDropIgnoreTimer = null
const clearedActivityEventIds = new Map()
const assistantStreamTargets = new Map()
const visualIndexRuns = new Set()
// Knowledge precipitation (Stream 1) — front-end orchestration. Serial queue so a
// bulk import never fires N concurrent LLM extractions; the card reads knowledgeByDoc.
const knowledgeByDoc = reactive({})
const precipitationQueue = []
const precipitationQueued = new Set() // doc ids currently queued or in-flight
let precipitationActive = null
let precipitationWatchdog = null // frees a stuck slot if its terminal event is lost
let assistantStreamDrainTimer = null

// ---------------------------------------------------------------------------
// Agent workspace sessions (Cursor-style)
//
// A session is a first-class conversation that is INDEPENDENT of the selected
// document. Chat messages live on the session (not on the document), so
// switching documents in the left pane never swaps or loses the conversation.
// Each session has a focus document (its default retrieval target) and is
// persisted in SQLite via the chat_sessions backend commands.
// ---------------------------------------------------------------------------
const chatSessions = reactive(new Map()) // id -> session object
const openSessionIds = usePersistedRef('openSessionIds', []) // tab order (ids)
const activeSessionId = usePersistedRef('activeSessionId', '')
const sessionsLoaded = ref(false)
const sessionHistoryOpen = ref(false)

// All sessions, most-recently-updated first (drives the 🕐 history list).
const sessionList = computed(() => Array.from(chatSessions.values())
  .slice()
  .sort((a, b) => (b.updatedAt || 0) - (a.updatedAt || 0)))
// Sessions currently shown as tabs (filtered to ones that still exist).
const openSessions = computed(() => openSessionIds.value
  .map((id) => chatSessions.get(id))
  .filter(Boolean))
const activeSession = computed(() => (
  activeSessionId.value ? chatSessions.get(activeSessionId.value) || null : null
))
// The activity event id of the in-flight generation in ANY open session, so the
// composer's stop button works even after the user navigates to a different
// paper/session while a generation is still streaming in the one they left.
const globalRunningEventId = computed(() => {
  for (const session of openSessions.value) {
    const messages = session?.messages || []
    const last = messages[messages.length - 1]
    if (last && last.role === 'assistant' && last.status === 'running') {
      return last.activityEventId || ''
    }
  }
  return ''
})
// The document a session's retrieval defaults to. Falls back to the viewed
// document so an as-yet-unfocused session still has somewhere to ask.
const activeFocusDoc = computed(() => {
  const session = activeSession.value
  if (session?.focusDocId) {
    const doc = allDocs.value.find((item) => item.id === session.focusDocId)
    if (doc) return doc
  }
  return selectedDocument.value
})

// Human label for a tab/history row: the LLM-generated title once present,
// otherwise a live truncation of the first question (the hybrid strategy), and
// finally a generic placeholder for an empty session.
function sessionTabTitle(session) {
  if (!session) return ui.value.newSession
  if (session.title) return session.title
  const firstUser = (session.messages || []).find((message) => message.role === 'user')
  const text = firstUser ? messageDisplayText(firstUser).trim() : ''
  if (text) return text.length > 20 ? `${text.slice(0, 20)}…` : text
  return ui.value.newSession
}

// Tab descriptors for ChatPane (kept out of the template for clarity).
const sessionTabs = computed(() => openSessions.value.map((session) => ({
  id: session.id,
  title: sessionTabTitle(session),
  active: session.id === activeSessionId.value,
})))
// History rows (all sessions) for the dropdown.
const sessionHistoryItems = computed(() => sessionList.value.map((session) => ({
  id: session.id,
  title: sessionTabTitle(session),
  focusTitle: session.focusDocTitle || '',
  turnCount: session.turnCount || 0,
  active: session.id === activeSessionId.value,
})))

// Merge a backend ChatSessionOutput into the local map, preserving runtime-only
// fields (messages, load flags) on an existing session.
function upsertSessionFromBackend(row) {
  if (!row || !row.id) return null
  const existing = chatSessions.get(row.id)
  if (existing) {
    if (row.title) {
      existing.title = row.title
      existing.titleGenerated = true
    }
    existing.focusDocId = row.focusDocumentId || existing.focusDocId
    existing.focusDocTitle = row.focusDocumentTitle || existing.focusDocTitle
    existing.referencedDocIds = row.referencedDocumentIds || existing.referencedDocIds
    if (typeof row.turnCount === 'number') existing.turnCount = row.turnCount
    existing.updatedAt = row.updatedAt || existing.updatedAt
    return existing
  }
  const session = {
    id: row.id,
    title: row.title || '',
    titleGenerated: Boolean(row.title),
    titleGenerating: false,
    focusDocId: row.focusDocumentId || '',
    focusDocTitle: row.focusDocumentTitle || '',
    referencedDocIds: Array.isArray(row.referencedDocumentIds) ? row.referencedDocumentIds : [],
    messages: [],
    chatHistoryLoaded: false,
    chatHistoryLoading: null,
    chatHistoryClearGeneration: 0,
    turnCount: row.turnCount || 0,
    createdAt: row.createdAt || 0,
    updatedAt: row.updatedAt || 0,
  }
  chatSessions.set(row.id, session)
  // Return the reactive proxy from the Map, not the raw object: callers mutate the
  // returned session (e.g. handleSend pushes messages), and mutating the raw object
  // bypasses reactivity, so the first message in a freshly-created session (common
  // on the Trending page, where no session is active yet) wouldn't render until a
  // later mutation forced a re-render.
  return chatSessions.get(row.id)
}

async function loadSessionList() {
  try {
    const rows = await invoke('list_chat_sessions')
    if (Array.isArray(rows)) rows.forEach(upsertSessionFromBackend)
  } catch (err) {
    console.warn('Failed to load chat sessions', err)
  } finally {
    sessionsLoaded.value = true
    reconcileSessionTabs()
  }
}

// Keep persisted tab/active state honest: drop ids that no longer exist and pick
// a sane active session when the persisted one is gone.
function reconcileSessionTabs() {
  let validOpen = openSessionIds.value.filter((id) => chatSessions.has(id))
  // First run after the migration: no tabs were ever persisted but sessions
  // exist (migrated per-document chats). Open the most recent so the user lands
  // on a populated conversation instead of a blank pane.
  if (!validOpen.length && sessionList.value.length) {
    validOpen = [sessionList.value[0].id]
  }
  if (validOpen.length !== openSessionIds.value.length
    || validOpen.some((id, i) => id !== openSessionIds.value[i])) {
    openSessionIds.value = validOpen
  }
  if (activeSessionId.value && !chatSessions.has(activeSessionId.value)) {
    activeSessionId.value = validOpen[0] || ''
  }
  if (!activeSessionId.value && validOpen.length) activeSessionId.value = validOpen[0]
  const session = activeSession.value
  if (session) void ensureSessionHistory(session)
}

async function createSession(focusDocId = '') {
  try {
    const focus = focusDocId && focusDocId !== 'empty' ? focusDocId : ''
    const row = await invoke('create_chat_session', {
      input: { focusDocumentId: focus || null, title: '' },
    })
    return upsertSessionFromBackend(row)
  } catch (err) {
    console.warn('Failed to create chat session', err)
    return null
  }
}

// Return the active session, creating one (focused on the current document) if
// none is active yet. This is the lazy-create path used on first send.
async function ensureActiveSession() {
  if (activeSession.value) return activeSession.value
  const session = await createSession(selectedDocId.value)
  if (!session) return null
  if (!openSessionIds.value.includes(session.id)) {
    openSessionIds.value = [...openSessionIds.value, session.id]
  }
  activeSessionId.value = session.id
  return session
}

// Retarget the active session's focus document (e.g. the user navigated to a
// different PDF and wants the conversation to now center on it). Updates locally
// then persists; the change is low-stakes if the backend write fails.
async function handleSetSessionFocus(docId) {
  const session = activeSession.value
  if (!session || !docId || docId === 'empty') return
  if (session.focusDocId === docId) return
  session.focusDocId = docId
  const doc = allDocs.value.find((item) => item.id === docId)
  if (doc) session.focusDocTitle = doc.shortTitle || doc.title || ''
  try {
    await invoke('update_chat_session_focus', {
      input: { id: session.id, focusDocumentId: docId },
    })
  } catch (err) {
    console.warn('Failed to update session focus', err)
  }
}

function setActiveSession(id) {
  if (!id || !chatSessions.has(id)) return
  if (!openSessionIds.value.includes(id)) {
    openSessionIds.value = [...openSessionIds.value, id]
  }
  activeSessionId.value = id
  sessionHistoryOpen.value = false
  const session = chatSessions.get(id)
  if (session) void ensureSessionHistory(session)
}

async function handleNewSession() {
  const session = await createSession(selectedDocId.value)
  if (!session) return
  openSessionIds.value = [...openSessionIds.value.filter((id) => id !== session.id), session.id]
  activeSessionId.value = session.id
  sessionHistoryOpen.value = false
}

function closeSessionTab(id) {
  const index = openSessionIds.value.indexOf(id)
  if (index === -1) return
  const next = openSessionIds.value.filter((item) => item !== id)
  openSessionIds.value = next
  if (activeSessionId.value === id) {
    activeSessionId.value = next[index] || next[index - 1] || ''
    const session = activeSession.value
    if (session) void ensureSessionHistory(session)
  }
}

async function deleteSessionById(id) {
  if (!id) return
  try {
    await invoke('delete_chat_session', { input: { id } })
  } catch (err) {
    console.warn('Failed to delete chat session', err)
    return
  }
  chatSessions.delete(id)
  closeSessionTab(id)
  if (activeSessionId.value === id) activeSessionId.value = ''
}

// Load persisted turns for a session once (idempotent, mirrors the old
// per-document loader but keyed by session_id).
async function ensureSessionHistory(session) {
  if (!session || !session.id || session.chatHistoryLoaded) return
  if (session.chatHistoryLoading) return session.chatHistoryLoading
  const clearGeneration = Number(session.chatHistoryClearGeneration || 0)
  session.chatHistoryLoading = (async () => {
    try {
      const history = await invoke('load_chat_turns', {
        input: { sessionId: session.id, limit: 60 },
      })
      if (Number(session.chatHistoryClearGeneration || 0) !== clearGeneration) return
      session.chatHistoryLoaded = true
      if (!Array.isArray(history) || !history.length) return
      const historyMessages = history.map((message) => ({
        id: message.id,
        sessionId: session.id,
        turnId: message.turnId || '',
        role: message.role,
        content: { en: message.content || '', zh: message.content || '' },
        provider: message.provider || '',
        reasoningContent: message.reasoningContent || '',
        citations: message.citations || [],
        claims: message.claims || [],
        retrievalTrace: message.retrievalTrace || null,
        activityEvents: message.retrievalTrace?.events || [],
        imageDataUrl: message.imageDataUrl || null,
        mentionedDocumentIds: message.referencedDocumentIds || [],
        status: 'succeeded',
        canContinueRetrieval: false,
        continuationRequest: null,
        persisted: true,
      }))
      mergeSessionHistoryMessages(session, historyMessages)
      hydrateHistoryRecommendations(session)
      if (session.id === activeSessionId.value) {
        activeCitationId.value = session.messages.flatMap((message) => message.citations || [])[0]?.id || ''
      }
    } catch (err) {
      session.chatHistoryLoaded = false
      console.warn('Failed to load session history', err)
    } finally {
      session.chatHistoryLoading = null
    }
  })()
  return session.chatHistoryLoading
}

function mergeSessionHistoryMessages(session, historyMessages) {
  const existingMessages = Array.isArray(session.messages) ? session.messages : []
  const welcomePrefix = `welcome-${session.id}`
  const onlyWelcome = existingMessages.length === 0
    || existingMessages.every((message) => String(message.id || '').startsWith(welcomePrefix))
  if (onlyWelcome) {
    session.messages = historyMessages
    return
  }
  const existingIds = new Set(existingMessages.map((message) => message.id))
  const existingFingerprints = new Set(existingMessages.map(messageFingerprint))
  const missingHistory = historyMessages.filter((message) => (
    !existingIds.has(message.id) && !existingFingerprints.has(messageFingerprint(message))
  ))
  session.messages = [...missingHistory, ...existingMessages]
}

// Hybrid title: after the first answer lands, ask the LLM for a <=12-char title
// and replace the temporary truncation. Best-effort; never blocks the answer.
async function maybeGenerateSessionTitle(session) {
  if (!session || session.titleGenerated || session.titleGenerating) return
  const hasUser = (session.messages || []).some((message) => message.role === 'user')
  const hasAnswer = (session.messages || []).some(
    (message) => message.role === 'assistant' && message.status === 'succeeded',
  )
  if (!hasUser || !hasAnswer) return
  session.titleGenerating = true
  try {
    const { providerId, modelKey } = parseChatModelOptionId(selectedChatModelId.value)
    const title = await invoke('generate_session_title', {
      input: {
        sessionId: session.id,
        modelProviderId: providerId || null,
        modelKey: modelKey || null,
      },
    })
    if (title) {
      session.title = title
      session.titleGenerated = true
    }
  } catch (err) {
    console.warn('Failed to generate session title', err)
  } finally {
    session.titleGenerating = false
  }
}

function chatStreamDebugEnabled() {
  return typeof window !== 'undefined'
    && window.localStorage?.getItem(CHAT_STREAM_DEBUG_STORAGE_KEY) === '1'
}

function chatStreamDebug(label, payload = {}) {
  if (!chatStreamDebugEnabled()) return
  console.debug(`[chat-stream] ${label}`, payload)
}

// Dedupe by id: a source can transiently appear under more than one storage
// root (legacy disk root + the Knowledge root after a re-import), but it is one
// document — the collection tree and search must not double-count it.
const allDocs = computed(() => {
  const seen = new Set()
  const out = []
  for (const workspaceRoot of workspace.roots) {
    for (const folder of workspaceRoot.folders) {
      for (const doc of folder.docs) {
        if (doc && doc.id && !seen.has(doc.id)) {
          seen.add(doc.id)
          out.push(doc)
        }
      }
    }
  }
  return out
})
// Locally-installed Codex / Claude as virtual chat models (no DB row — computed
// from detection). The synthetic provider id `local-agent-<kind>` is recognized
// by the backend dispatch; modelKey is a placeholder.
const localAgentChatModels = computed(() => (
  localAgentStatus.value
    .filter((agent) => agent.installed)
    .map((agent) => ({
      id: makeChatModelOptionId(`local-agent-${agent.kind}`, 'default'),
      providerId: `local-agent-${agent.kind}`,
      provider: agent.label,
      providerType: 'local-agent',
      modelKey: 'default',
      modelId: agent.kind,
      label: `${agent.label} (local)`,
      // Codex takes images via `codex exec -i`; Claude's headless `-p` has no image
      // flag, so it stays text-only. This flag gates the chat composer's paste/attach.
      capabilities: agent.kind === 'codex' ? ['text', 'vision'] : ['text'],
    }))
))
const configuredChatModels = computed(() => ([
  ...modelProviders.value
    .flatMap((provider) => (
      (provider.models || [])
        .filter((model) => model.enabled)
        .map((model) => ({
          id: makeChatModelOptionId(provider.id, model.key),
          providerId: provider.id,
          provider: provider.name,
          providerType: provider.providerType,
          modelKey: model.key,
          modelId: model.modelId,
          label: model.nickname || model.modelId,
          capabilities: model.capabilities?.length ? model.capabilities : ['text'],
        }))
    )),
  ...localAgentChatModels.value,
]))
const chatModelConfigured = computed(() => configuredChatModels.value.length > 0)
const availableChatModels = computed(() => (
  chatModelConfigured.value
    ? configuredChatModels.value
    : [{
      id: UNCONFIGURED_CHAT_MODEL_ID,
      provider: '',
      label: ui.value.modelNotConfigured,
      capabilities: ['text'],
      disabled: true,
    }]
))
const emptyDocument = computed(() => ({
  id: 'empty',
  title: ui.value.noDocumentSelected,
  shortTitle: ui.value.noDocumentSelected,
  status: 'stale',
  statusTone: 'danger',
  treeReady: false,
  lastOpened: { en: '', zh: '' },
  pageCount: 0,
  indexVersion: 0,
  currentIndexVersion: 0,
  visualIndexStatus: 'pending',
  visualIndexVersion: 0,
  visualIndexError: '',
  indexProgress: {
    percent: 0,
    stage: '',
    label: '',
  },
  currentPage: 1,
  chatModelId: UNCONFIGURED_CHAT_MODEL_ID,
  quoteBlockId: '',
  chatReady: false,
  translation: {
    status: 'idle',
    progress: 0,
    total: 0,
    failedBlocks: 0,
    lang: translationLang.value,
    error: '',
    jobId: '',
    providerKey: '',
    phase: '',
    currentPage: 0,
    pdfJobId: '',
    pdfStatus: 'idle',
    pdfProgressPercent: 0,
    monoPdfPath: '',
    dualPdfPath: '',
    pdfArtifactScope: '',
    pdfArtifactPages: '',
    partialArtifacts: {},
    cached: false,
    pages: {},
  },
  pages: [],
  messages: [],
  notes: [],
  notesLoaded: false,
  notesLoading: null,
}))
// Knowledge-base pivot: an empty (or stale) selection resolves to the empty
// document so the center shows the "ask my knowledge base" home — NOT the first
// PDF. Opening a source is always an explicit act now.
const selectedDocument = computed(() => (
  allDocs.value.find((doc) => doc.id === selectedDocId.value)
  || emptyDocument.value
))

// Knowledge-base pivot (P2): authored text sources open in the Markdown editor.
const EDITABLE_CONTENT_TYPES = ['note', 'markdown', 'text', 'web']
function isEditableSourceDoc(doc) {
  return Boolean(doc) && doc.id !== 'empty' && EDITABLE_CONTENT_TYPES.includes(doc.contentType)
}
const editableSelected = computed(() => isEditableSourceDoc(selectedDocument.value))
// Knowledge-base pivot (P3): Office sources render in the client-side previewer.
const OFFICE_CONTENT_TYPES = ['docx', 'xlsx', 'pptx']
const officeSelected = computed(() => (
  selectedDocument.value
  && selectedDocument.value.id !== 'empty'
  && OFFICE_CONTENT_TYPES.includes(selectedDocument.value.contentType)
))
// True while a web clip is being fetched (gates the home clip form).
const clipBusy = ref(false)
// Starter questions shown in the chat empty state. Two layers, no extra LLM call:
// (A) static common questions, always available even before indexing finishes; and
// (B) document-specific prompts derived from the already-computed knowledge card's
// top concepts. Empty when there's no real document open.
const suggestedQuestions = computed(() => {
  if (!selectedDocument.value || selectedDocument.value.id === 'empty') return []
  const t = ui.value
  const questions = [
    t.suggestQSummary,
    t.suggestQContributions,
    t.suggestQMethods,
    t.suggestQLimitations,
  ]
  const concepts = Array.isArray(knowledgeCard.value?.concepts)
    ? knowledgeCard.value.concepts
    : []
  for (const concept of concepts.slice(0, 2)) {
    const name = String(concept?.name || '').trim()
    if (name) questions.push(t.suggestQExplainConcept.replace('{concept}', name))
  }
  return questions.filter(Boolean)
})
// Show the Trending discovery feed ONLY when the user explicitly opens it. It is
// a secondary surface now (knowledge-base pivot) — the empty state defaults to the
// conversation, never to Trending. Gated off entirely when discovery is disabled.
const showTrending = computed(() => (
  trendingEnabled.value && trendingView.value
))
// No document open (and not on a discovery surface) → the main stage IS the
// conversation. The single ChatPane fills the center; there is no separate reader
// column and no duplicate right-side chat.
const conversationCentered = computed(() => (
  !showTrending.value && !showGraph.value && selectedDocument.value.id === 'empty'
))
// Tab descriptors for the reader tab bar: resolve each open id to its document,
// dropping any that no longer exist. Status mirrors the sidebar status dot.
const activeWorkspaceRoot = computed(() => {
  if (!workspace.roots.length) return null
  const selectedId = selectedDocId.value
  return workspace.roots.find((workspaceRoot) => (
    workspaceRoot.folders.some((folder) => folder.docs.some((doc) => doc.id === selectedId))
  )) || workspace.roots[0]
})
const currentChatModel = computed(() => {
  return availableChatModels.value.find((model) => model.id === selectedChatModelId.value)
    || availableChatModels.value[0]
    || null
})
const translationProviderNote = computed(() => {
  if (translationProvider.value === 'google-web') return ui.value.translationProviderGoogleWebNote
  if (translationProvider.value === 'microsoft') return ui.value.translationProviderMicrosoftNote
  if (translationProvider.value === 'llm') return ui.value.translationProviderLlmNote
  return ui.value.translationProviderPlaceholderNote
})
const hasModelProviderDraft = computed(() => (
  Boolean(providerForm.id)
  || providerForm.models.some((model) => String(model.modelId || '').trim())
  || Boolean(String(providerForm.apiKey || '').trim())
))
const providerConnectionSummary = computed(() => {
  const keyState = providerForm.hasApiKey
    ? ui.value.apiKeySaved
    : ui.value.apiKeyNotSaved
  return `${providerForm.name || ui.value.newProvider} · ${providerForm.baseUrl || providerTypePreset.value.baseUrl} · ${keyState}`
})

function preferredLocalAgentModelId() {
  // Preferred local agent first, then the other; only installed ones are listed.
  const order = preferredLocalAgent.value === 'claude' ? ['claude', 'codex'] : ['codex', 'claude']
  for (const kind of order) {
    const match = localAgentChatModels.value.find((model) => model.modelId === kind)
    if (match) return match.id
  }
  return ''
}

function defaultConfiguredChatModelId() {
  if (!chatModelConfigured.value) return UNCONFIGURED_CHAT_MODEL_ID
  // Prefer the user's last explicit pick if it's still a configured/enabled model.
  if (lastChatModelId.value && configuredChatModels.value.some((model) => model.id === lastChatModelId.value)) {
    return lastChatModelId.value
  }
  // A configured HTTP provider wins over the local-agent zero-config default.
  const hasEnabledModel = (provider) => (provider?.models || []).some((model) => model.enabled)
  const defaultProvider = modelProviders.value.find((provider) => provider.isDefault && hasEnabledModel(provider))
    || modelProviders.value.find(hasEnabledModel)
  const defaultModelId = defaultProvider ? defaultChatModelOptionId(defaultProvider) : ''
  if (defaultModelId && configuredChatModels.value.some((model) => model.id === defaultModelId)) {
    return defaultModelId
  }
  // No HTTP provider configured → fall back to an installed local agent (the
  // zero-config first-run path: Codex → Claude by default).
  const localId = preferredLocalAgentModelId()
  if (localId) return localId
  return configuredChatModels.value[0]?.id || UNCONFIGURED_CHAT_MODEL_ID
}

function resolveChatModelId(modelId) {
  const candidate = String(modelId || '')
  if (availableChatModels.value.some((model) => model.id === candidate)) return candidate
  return defaultConfiguredChatModelId()
}

function applySelectedChatModel(modelId, doc = selectedDocument.value) {
  const nextModelId = resolveChatModelId(modelId)
  selectedChatModelId.value = nextModelId
  if (doc && doc.id !== 'empty') doc.chatModelId = nextModelId
  return nextModelId
}

watch(selectedDocument, (doc) => {
  if (!doc) return
  syncingTranslationLangFromDocument = true
  translationLang.value = doc.translation.lang || 'zh'
  nextTick(() => {
    syncingTranslationLangFromDocument = false
  })
  viewMode.value = 'original'
  activePage.value = doc.currentPage || doc.pages[0]?.page || 1
  activeBlockId.value = doc.quoteBlockId || doc.pages[0]?.blocks?.[0]?.id || ''
  activeCitationId.value = activeSession.value?.messages?.flatMap((message) => message.citations || [])[0]?.id || ''
  activeHighlight.value = null
  hoveredLinkedBlock.value = null
  lastSelection.value = null
  activeTranslation.value = null
  inlineTranslateOpen.value = false
  activeNoteId.value = ''
  noteComposer.value = { ...noteComposer.value, open: false }
  closeNoteDeleteConfirm()
  applySelectedChatModel(doc.chatModelId, doc)
  loadNotesForDocument(doc.id)
  scheduleIdleTask(() => scheduleDocumentVisualIndex(doc), 1800)
  // Re-apply a cross-document citation jump after the reader-state reset above,
  // so the target page/highlight wins over the doc's saved currentPage. Always
  // consume the pending jump on the first activation change: if the target doc
  // loaded we apply it, otherwise (e.g. the doc was deleted and we fell back to
  // another) we discard it so it can't fire later on an unrelated activation.
  if (pendingCitationJump) {
    const jump = pendingCitationJump
    pendingCitationJump = null
    if (jump.documentId === doc.id) {
      nextTick(() => applyCitationJump(jump))
    }
  }
}, { immediate: true })

watch(translationLang, (lang, previousLang) => {
  const doc = selectedDocument.value
  if (!doc) return
  if (syncingTranslationLangFromDocument) {
    doc.translation.lang = lang
    return
  }
  if (previousLang && previousLang !== lang) {
    resetDocumentTranslationState(doc, lang)
    if (['translated', 'dual'].includes(viewMode.value)) viewMode.value = 'original'
    return
  }
  doc.translation.lang = lang
})

watch(translationProvider, () => {
  const doc = selectedDocument.value
  if (!doc || doc.id === 'empty') return
  resetDocumentTranslationState(doc, translationLang.value)
  if (['translated', 'dual'].includes(viewMode.value)) viewMode.value = 'original'
})

watch([selectedDocId, activePage, translationLang, viewMode], () => {
  if (viewMode.value !== 'dual') hoveredLinkedBlock.value = null
  if (['translated', 'dual'].includes(viewMode.value)) {
    scheduleActivePageTranslationLoad(0)
  }
})

watch(selectedChatModelId, (modelId) => {
  const nextModelId = resolveChatModelId(modelId)
  if (nextModelId !== modelId) {
    selectedChatModelId.value = nextModelId
    return
  }
  if (selectedDocument.value && selectedDocument.value.id !== 'empty') {
    selectedDocument.value.chatModelId = nextModelId
  }
})

// Persist the model ONLY when the user explicitly picks one (not on the
// programmatic per-document restore that runs while switching documents —
// otherwise opening a doc that carries its own model would clobber the user's
// last deliberate choice).
function handleSelectChatModel(modelId) {
  selectedChatModelId.value = modelId
  const resolved = resolveChatModelId(modelId)
  if (resolved && resolved !== UNCONFIGURED_CHAT_MODEL_ID) {
    lastChatModelId.value = resolved
  }
}

watch(availableChatModels, () => {
  // A late-arriving option (local agents finish detecting ~after startup) can make
  // the user's persisted pick selectable only now. If we're sitting on a fallback
  // (e.g. an HTTP default chosen before detection landed), restore the saved pick.
  // Safe against in-session choices: an explicit pick keeps lastChatModelId in sync
  // with the selection, so this branch only fires for programmatic fallbacks.
  if (
    lastChatModelId.value &&
    lastChatModelId.value !== selectedChatModelId.value &&
    configuredChatModels.value.some((model) => model.id === lastChatModelId.value)
  ) {
    applySelectedChatModel(lastChatModelId.value)
    return
  }
  applySelectedChatModel(selectedDocument.value?.chatModelId || selectedChatModelId.value)
})

watch(() => providerForm.providerType, (providerType) => {
  if (providerTypeSyncSuspended) return
  applyProviderTypePreset(providerType)
}, { flush: 'sync' })

const providerTypePreset = computed(() => (
  MODEL_PROVIDER_PRESETS[providerForm.providerType] || MODEL_PROVIDER_PRESETS['openai-compatible']
))

function createEmptyProviderForm() {
  const preset = MODEL_PROVIDER_PRESETS['openai-compatible']
  return {
    id: '',
    name: preset.name,
    providerType: 'openai-compatible',
    baseUrl: preset.baseUrl,
    models: [createProviderModelDraft('openai-compatible')],
    defaultModelKey: '',
    apiKey: '',
    enabled: true,
    isDefault: true,
    hasApiKey: false,
  }
}

function createProviderModelDraft(providerType = 'openai-compatible', model = null) {
  const preset = MODEL_PROVIDER_PRESETS[providerType] || MODEL_PROVIDER_PRESETS['openai-compatible']
  const modelId = model?.modelId ?? preset.model ?? ''
  return {
    key: model?.key || `draft-model-${localModelDraftCounter++}`,
    modelId,
    nickname: model?.nickname ?? modelId,
    capabilities: normalizeCapabilities(model?.capabilities ?? inferModelCapabilities(providerType, modelId)),
    enabled: model?.enabled ?? true,
    isDefaultChatModel: model?.isDefaultChatModel ?? true,
    // Context window: user override (authoritative) + value auto-detected from
    // the provider's /models endpoint. Empty override = "auto".
    contextWindowOverride: model?.contextWindowOverride ?? null,
    detectedContextWindow: model?.detectedContextWindow ?? null,
  }
}

function createEmptyMicrosoftForm() {
  return {
    endpoint: 'https://api.cognitive.microsofttranslator.com',
    region: '',
    apiKey: '',
    hasApiKey: false,
  }
}

function resetProviderForm(provider = null) {
  const next = provider || createEmptyProviderForm()
  providerTypeSyncSuspended = true
  providerForm.id = next.id || ''
  providerForm.name = next.name || 'OpenAI Compatible'
  providerForm.providerType = next.providerType || 'openai-compatible'
  providerForm.baseUrl = next.baseUrl || 'https://api.openai.com/v1'
  providerForm.models = ((next.models && next.models.length) ? next.models : [createProviderModelDraft(next.providerType || 'openai-compatible')])
    .map((model) => createProviderModelDraft(next.providerType || 'openai-compatible', model))
  providerForm.defaultModelKey = next.defaultModelKey || providerForm.models.find((model) => model.isDefaultChatModel)?.key || providerForm.models[0]?.key || ''
  providerForm.apiKey = ''
  providerForm.enabled = true
  providerForm.isDefault = next.isDefault ?? true
  providerForm.hasApiKey = Boolean(next.hasApiKey)
  providerTypeSyncSuspended = false
}

function providerEditKey(provider) {
  return provider?._editKey || (provider?.id ? `saved:${provider.id}` : '')
}

function cloneProviderForEdit(provider, editKey = '') {
  const providerType = provider?.providerType || 'openai-compatible'
  const next = provider || createEmptyProviderForm()
  return {
    _editKey: editKey || providerEditKey(provider) || `draft-provider-${localProviderDraftCounter++}`,
    id: next.id || '',
    name: next.name || 'OpenAI Compatible',
    providerType,
    baseUrl: next.baseUrl || 'https://api.openai.com/v1',
    models: ((next.models && next.models.length) ? next.models : [createProviderModelDraft(providerType)])
      .map((model) => createProviderModelDraft(providerType, model)),
    defaultModelKey: next.defaultModelKey || next.models?.find((model) => model.isDefaultChatModel)?.key || next.models?.[0]?.key || '',
    apiKey: next.apiKey || '',
    enabled: true,
    isDefault: next.isDefault ?? false,
    hasApiKey: Boolean(next.hasApiKey),
  }
}

function snapshotProviderForm(editKey = selectedProviderEditKey.value) {
  return {
    _editKey: editKey,
    id: providerForm.id || '',
    name: providerForm.name,
    providerType: providerForm.providerType,
    baseUrl: providerForm.baseUrl,
    models: providerForm.models.map((model) => ({
      key: model.key,
      modelId: model.modelId,
      nickname: model.nickname,
      capabilities: [...normalizeCapabilities(model.capabilities)],
      enabled: model.enabled,
      isDefaultChatModel: model.key === providerForm.defaultModelKey,
      contextWindowOverride: normalizeContextWindow(model.contextWindowOverride),
      detectedContextWindow: normalizeContextWindow(model.detectedContextWindow),
    })),
    defaultModelKey: providerForm.defaultModelKey,
    apiKey: providerForm.apiKey,
    enabled: true,
    isDefault: providerForm.isDefault,
    hasApiKey: providerForm.hasApiKey,
  }
}

function persistCurrentProviderEdit() {
  if (!selectedProviderEditKey.value) return
  const index = editableProviders.value.findIndex((provider) => providerEditKey(provider) === selectedProviderEditKey.value)
  if (index === -1) return
  editableProviders.value[index] = snapshotProviderForm()
}

function initializeEditableProviders() {
  editableProviders.value = modelProviders.value.map((provider) => cloneProviderForEdit(provider, `saved:${provider.id}`))
  let selected = editableProviders.value.find((provider) => provider.isDefault) || editableProviders.value[0]
  if (!selected) {
    selected = cloneProviderForEdit(createEmptyProviderForm(), `draft-provider-${localProviderDraftCounter++}`)
    selected.name = `${ui.value.newProvider} 1`
    selected.isDefault = true
    editableProviders.value.push(selected)
  } else if (!editableProviders.value.some((provider) => provider.isDefault)) {
    selected.isDefault = true
  }
  selectedProviderEditKey.value = providerEditKey(selected)
  resetProviderForm(selected)
}

function providerListName(provider) {
  return providerEditKey(provider) === selectedProviderEditKey.value
    ? providerForm.name || ui.value.newProvider
    : provider.name || ui.value.newProvider
}

function providerListMeta(provider) {
  if (!provider.id) return ui.value.unsavedProvider
  return `${provider.providerType} · ${providerModelCountLabel(provider)}`
}

function ensureProviderSelectionAfterRemoval(preferredKey = '') {
  let next = editableProviders.value.find((provider) => providerEditKey(provider) === preferredKey)
    || editableProviders.value.find((provider) => provider.isDefault)
    || editableProviders.value[0]
  if (!next) {
    next = cloneProviderForEdit(createEmptyProviderForm(), `draft-provider-${localProviderDraftCounter++}`)
    next.name = `${ui.value.newProvider} 1`
    next.isDefault = true
    editableProviders.value = [next]
  }
  if (!editableProviders.value.some((provider) => provider.isDefault)) {
    next.isDefault = true
  }
  selectedProviderEditKey.value = providerEditKey(next)
  resetProviderForm(next)
}

function removeEditableProvider(provider) {
  const key = providerEditKey(provider)
  if (!key || settingsStatus.value === 'saving') return
  persistCurrentProviderEdit()

  if (provider.id) {
    // Persisted provider — confirm through the in-app modal. window.confirm()
    // silently returns false in Tauri's WKWebView, which made delete a no-op.
    providerDeleteError.value = ''
    providerDeleteStatus.value = 'idle'
    providerDeleteTarget.value = { key, id: provider.id, name: providerListName(provider) }
    providerDeleteConfirmOpen.value = true
    return
  }

  // Unsaved draft — nothing persisted yet, so drop it locally without a prompt.
  editableProviders.value = editableProviders.value.filter((item) => providerEditKey(item) !== key)
  if (selectedProviderEditKey.value !== key) return
  ensureProviderSelectionAfterRemoval()
}

function closeDeleteProviderConfirm() {
  if (providerDeleteStatus.value === 'deleting') return
  providerDeleteConfirmOpen.value = false
  providerDeleteStatus.value = 'idle'
  providerDeleteError.value = ''
  providerDeleteTarget.value = { key: '', id: '', name: '' }
}

function confirmDeleteProvider() {
  const target = providerDeleteTarget.value
  if (!target?.id || providerDeleteStatus.value === 'deleting') return
  providerDeleteStatus.value = 'deleting'
  providerDeleteError.value = ''
  settingsStatus.value = 'saving'
  settingsError.value = ''
  providerTestStatus.value = 'idle'
  providerTestMessage.value = ''
  modelFetchStatus.value = 'idle'
  modelFetchMessage.value = ''

  void (async () => {
    try {
      await invoke('delete_model_provider', { input: { id: target.id } })
      const preferredKey = selectedProviderEditKey.value === target.key ? '' : selectedProviderEditKey.value
      await loadModelProviders()
      const unsavedDrafts = editableProviders.value
        .filter((item) => !item.id && providerEditKey(item) !== target.key)
      editableProviders.value = [
        ...modelProviders.value.map((item) => cloneProviderForEdit(item, `saved:${item.id}`)),
        ...unsavedDrafts,
      ]
      ensureProviderSelectionAfterRemoval(preferredKey)
      settingsStatus.value = 'saved'
      providerDeleteConfirmOpen.value = false
      providerDeleteStatus.value = 'idle'
      providerDeleteTarget.value = { key: '', id: '', name: '' }
    } catch (err) {
      settingsStatus.value = 'failed'
      settingsError.value = err?.message || String(err)
      providerDeleteStatus.value = 'failed'
      providerDeleteError.value = err?.message || String(err)
    }
  })()
}

function applyProviderTypePreset(providerType) {
  const preset = MODEL_PROVIDER_PRESETS[providerType] || MODEL_PROVIDER_PRESETS['openai-compatible']
  providerForm.name = preset.name
  providerForm.baseUrl = preset.baseUrl
  if (providerForm.models.length === 1) {
    const firstModel = providerForm.models[0]
    firstModel.modelId = preset.model
    firstModel.nickname = preset.model || ''
    firstModel.capabilities = normalizeCapabilities(inferModelCapabilities(providerType, preset.model || ''))
  }
}

function resetMicrosoftForm(settings = null) {
  microsoftForm.endpoint = settings?.microsoftEndpoint || 'https://api.cognitive.microsofttranslator.com'
  microsoftForm.region = settings?.microsoftRegion || ''
  microsoftForm.apiKey = ''
  microsoftForm.hasApiKey = Boolean(settings?.microsoftHasApiKey)
}

async function refreshLocalAgentStatus() {
  localAgentChecking.value = true
  try {
    const status = await invoke('get_local_agent_status')
    localAgentStatus.value = Array.isArray(status) ? status : []
  } catch (err) {
    console.warn('Failed to detect local agents', err)
    localAgentStatus.value = []
  } finally {
    localAgentChecking.value = false
  }
}

async function testLocalAgent(kind) {
  const providerId = `local-agent-${kind}`
  localAgentTest.value = {
    ...localAgentTest.value,
    [kind]: { testing: true, ok: null, message: ui.value.localAgentTesting },
  }
  try {
    const res = await invoke('test_local_agent_connection', { providerId })
    localAgentTest.value = {
      ...localAgentTest.value,
      [kind]: { testing: false, ok: Boolean(res?.ok), message: res?.message || '' },
    }
  } catch (err) {
    localAgentTest.value = {
      ...localAgentTest.value,
      [kind]: { testing: false, ok: false, message: String(err) },
    }
  }
}

async function loadModelProviders() {
  try {
    modelProviders.value = await invoke('list_model_providers')
    const defaultProvider = modelProviders.value.find((provider) => provider.isDefault)
      || modelProviders.value[0]
    if (defaultProvider) resetProviderForm(defaultProvider)
    // Resolve through defaultConfiguredChatModelId (last pick → HTTP default →
    // installed local agent → unconfigured). When local detection lands later,
    // the availableChatModels watch re-applies this.
    selectedChatModelId.value = defaultConfiguredChatModelId()
  } catch (err) {
    console.warn('Failed to load model providers', err)
  }
}

function defaultChatModelOptionId(provider) {
  const defaultModel = (provider.models || []).find((model) => model.key === provider.defaultModelKey && model.enabled)
    || (provider.models || []).find((model) => model.enabled)
  return defaultModel ? makeChatModelOptionId(provider.id, defaultModel.key) : UNCONFIGURED_CHAT_MODEL_ID
}

function makeChatModelOptionId(providerId, modelKey) {
  return `${providerId}::${modelKey}`
}

function parseChatModelOptionId(value) {
  const [providerId, modelKey] = String(value || '').split('::')
  return {
    providerId: providerId || '',
    modelKey: modelKey || '',
  }
}

function normalizeCapabilities(capabilities) {
  const next = Array.isArray(capabilities) ? capabilities.map((capability) => String(capability).trim().toLowerCase()).filter(Boolean) : []
  const unique = [...new Set(['text', ...next.filter((capability) => MODEL_CAPABILITY_OPTIONS.includes(capability))])]
  return unique
}

function inferModelCapabilities(providerType, modelId) {
  const normalized = String(modelId || '').toLowerCase()
  const capabilities = []
  if (
    normalized.includes('gpt-4o')
    || normalized.includes('gpt-4.1')
    || normalized.includes('gpt-5')
    || normalized.includes('o3')
    || normalized.includes('o4')
    || normalized.includes('claude-3')
    || normalized.includes('claude-4')
    || normalized.includes('gemini')
    || normalized.includes('vision')
    || normalized.includes('vl')
    || normalized.includes('pixtral')
  ) {
    capabilities.push('vision')
  }
  if (
    normalized.includes('reason')
    || normalized.includes('reasoner')
    || normalized.includes('thinking')
    || normalized.includes('o3')
    || normalized.includes('o4')
    || normalized.includes('gpt-5')
  ) {
    capabilities.push('reasoning')
  }
  if (['openai', 'openrouter', 'openai-compatible'].includes(providerType) || normalized.includes('deepseek')) {
    capabilities.push('tool_use')
  }
  return capabilities
}

function isEmptyProviderModelDraft(model) {
  return !String(model?.modelId || '').trim()
    && !String(model?.nickname || '').trim()
}

// Coerce a context-window field to a positive integer or null ("auto").
function normalizeContextWindow(value) {
  const parsed = Number(value)
  return Number.isFinite(parsed) && parsed >= 1024 ? Math.floor(parsed) : null
}

// Placeholder for the context-window override input: shows the auto-detected
// window (from /models) when known, otherwise just "auto".
function contextWindowPlaceholder(model) {
  const detected = normalizeContextWindow(model?.detectedContextWindow)
  return detected ? `${detected} · ${ui.value.contextWindowAuto}` : ui.value.contextWindowAuto
}

function mergeFetchedProviderModels(modelIds, contextWindows = {}) {
  // Refresh the auto-detected window on models we already have (the server's
  // /models value may have changed); never clobber a user override.
  providerForm.models.forEach((model) => {
    const detected = normalizeContextWindow(contextWindows[model.modelId])
    if (detected) model.detectedContextWindow = detected
  })
  const existingIds = new Set(
    providerForm.models
      .map((model) => String(model.modelId || '').trim().toLowerCase())
      .filter(Boolean),
  )
  const imported = []
  modelIds.forEach((modelId) => {
    const normalizedModelId = String(modelId || '').trim()
    if (!normalizedModelId) return
    const dedupeKey = normalizedModelId.toLowerCase()
    if (existingIds.has(dedupeKey)) return
    existingIds.add(dedupeKey)
    imported.push(createProviderModelDraft(providerForm.providerType, {
      modelId: normalizedModelId,
      nickname: normalizedModelId,
      capabilities: inferModelCapabilities(providerForm.providerType, normalizedModelId),
      enabled: true,
      isDefaultChatModel: false,
      detectedContextWindow: normalizeContextWindow(contextWindows[normalizedModelId]),
    }))
  })

  if (!imported.length) return 0
  const onlyEmptyPlaceholder = providerForm.models.length === 1 && isEmptyProviderModelDraft(providerForm.models[0])
  providerForm.models = onlyEmptyPlaceholder
    ? imported
    : [...providerForm.models, ...imported]
  if (!providerForm.defaultModelKey || onlyEmptyPlaceholder) {
    providerForm.defaultModelKey = providerForm.models[0]?.key || ''
  }
  setDefaultProviderModel(providerForm.models.find((model) => model.key === providerForm.defaultModelKey) || providerForm.models[0])
  return imported.length
}

async function fetchProviderModels() {
  modelFetchStatus.value = 'fetching'
  modelFetchMessage.value = ''
  settingsStatus.value = 'idle'
  settingsError.value = ''
  providerTestStatus.value = 'idle'
  providerTestMessage.value = ''
  try {
    const result = await invoke('fetch_provider_models', {
      input: {
        id: providerForm.id || null,
        providerType: providerForm.providerType,
        baseUrl: providerForm.baseUrl,
        apiKey: providerForm.apiKey || null,
      },
    })
    const importedCount = mergeFetchedProviderModels(result?.modelIds || [], result?.contextWindows || {})
    persistCurrentProviderEdit()
    modelFetchStatus.value = 'succeeded'
    modelFetchMessage.value = importedCount > 0
      ? ui.value.fetchModelsImported.replace('{count}', importedCount)
      : ui.value.fetchModelsNoNewModels
  } catch (err) {
    modelFetchStatus.value = 'failed'
    modelFetchMessage.value = err?.message || String(err)
  }
}

function addProviderModel() {
  const model = createProviderModelDraft(providerForm.providerType)
  model.isDefaultChatModel = false
  providerForm.models.push(model)
  if (!providerForm.defaultModelKey) {
    providerForm.defaultModelKey = providerForm.models.at(-1)?.key || ''
  }
}

function removeProviderModel(index) {
  if (providerForm.models.length === 1) return
  const removed = providerForm.models[index]
  providerForm.models.splice(index, 1)
  if (providerForm.defaultModelKey === removed?.key) {
    providerForm.defaultModelKey = providerForm.models[0]?.key || ''
  }
}

function setDefaultProviderModel(model) {
  providerForm.defaultModelKey = model.key
  providerForm.models.forEach((item) => {
    item.isDefaultChatModel = item.key === model.key
  })
}

function toggleModelCapability(model, capability) {
  const current = normalizeCapabilities(model.capabilities)
  if (current.includes(capability)) {
    model.capabilities = current.filter((item) => item !== capability)
  } else {
    model.capabilities = normalizeCapabilities([...current, capability])
  }
}

function modelCapabilityLabel(capability) {
  if (capability === 'vision') return ui.value.capabilityVision
  if (capability === 'reasoning') return ui.value.capabilityReasoning
  if (capability === 'tool_use') return ui.value.capabilityToolUse
  return capability
}

async function loadTranslationSettings() {
  try {
    const settings = await invoke('load_translation_settings')
    translationProvider.value = settings.provider || 'google-web'
    translationFallbackEnabled.value = settings.enableFallback ?? true
    resetMicrosoftForm(settings)
  } catch (err) {
    console.warn('Failed to load translation settings', err)
  }
}

async function loadWebSearchSettings() {
  try {
    const settings = await invoke('load_web_search_settings')
    webSearchForm.apiKey = ''
    webSearchForm.hasApiKey = Boolean(settings?.exaHasApiKey)
    webSearchForm.status = 'idle'
  } catch (err) {
    console.warn('Failed to load web search settings', err)
  }
}

async function saveWebSearchSettings() {
  webSearchForm.status = 'saving'
  try {
    const settings = await invoke('save_web_search_settings', {
      input: { exaApiKey: webSearchForm.apiKey },
    })
    webSearchForm.apiKey = ''
    webSearchForm.hasApiKey = Boolean(settings?.exaHasApiKey)
    webSearchForm.status = 'saved'
  } catch (err) {
    console.warn('Failed to save web search settings', err)
    webSearchForm.status = 'error'
  }
}

async function loadProxySettings() {
  try {
    const settings = await invoke('load_proxy_settings')
    proxyForm.proxyUrl = settings?.proxyUrl || ''
    proxyForm.status = 'idle'
  } catch (err) {
    console.warn('Failed to load proxy settings', err)
  }
}

async function saveProxySettings() {
  proxyForm.status = 'saving'
  try {
    const settings = await invoke('save_proxy_settings', {
      input: { proxyUrl: proxyForm.proxyUrl },
    })
    proxyForm.proxyUrl = settings?.proxyUrl || ''
    proxyForm.status = 'saved'
  } catch (err) {
    console.warn('Failed to save proxy settings', err)
    proxyForm.status = 'error'
  }
}

async function openSettings() {
  settingsOpen.value = true
  settingsSection.value = 'chat'
  settingsStatus.value = 'idle'
  settingsError.value = ''
  providerTestStatus.value = 'idle'
  providerTestMessage.value = ''
  modelFetchStatus.value = 'idle'
  modelFetchMessage.value = ''
  await loadWebSearchSettings()
  await loadProxySettings()
  await loadVaultSettings()
  await loadBackupSettings()
  await loadKnowledgeApi()
  await loadTranslationSettings()
  initializeEditableProviders()
}

function closeSettings() {
  settingsOpen.value = false
}

function switchSettingsSection(section) {
  persistCurrentProviderEdit()
  settingsSection.value = section
  settingsStatus.value = 'idle'
  settingsError.value = ''
  providerTestStatus.value = 'idle'
  providerTestMessage.value = ''
  modelFetchStatus.value = 'idle'
  modelFetchMessage.value = ''
}

function providerModelCount(provider) {
  return (provider.models || []).filter((model) => model.enabled).length
}

function providerModelCountLabel(provider) {
  const count = providerModelCount(provider)
  return count === 1 ? ui.value.oneModel : `${count} ${ui.value.modelsCount}`
}

function selectModelProvider(provider) {
  const nextKey = providerEditKey(provider)
  if (nextKey === selectedProviderEditKey.value) return
  persistCurrentProviderEdit()
  settingsStatus.value = 'idle'
  settingsError.value = ''
  providerTestStatus.value = 'idle'
  providerTestMessage.value = ''
  modelFetchStatus.value = 'idle'
  modelFetchMessage.value = ''
  selectedProviderEditKey.value = nextKey
  resetProviderForm(provider)
}

function setDefaultModelProvider(provider) {
  const nextKey = providerEditKey(provider)
  if (!nextKey || settingsStatus.value === 'saving') return
  persistCurrentProviderEdit()
  editableProviders.value = editableProviders.value.map((item) => ({
    ...item,
    isDefault: providerEditKey(item) === nextKey,
    enabled: true,
  }))
  const selected = editableProviders.value.find((item) => providerEditKey(item) === nextKey)
  if (!selected) return
  selectedProviderEditKey.value = nextKey
  resetProviderForm(selected)
  settingsStatus.value = 'idle'
  settingsError.value = ''
}

function createNewProvider() {
  persistCurrentProviderEdit()
  const next = createEmptyProviderForm()
  const draftNumber = localProviderDraftCounter++
  next._editKey = `draft-provider-${draftNumber}`
  next.name = `${ui.value.newProvider} ${draftNumber}`
  next.isDefault = editableProviders.value.length === 0
  const draft = cloneProviderForEdit(next, next._editKey)
  editableProviders.value.push(draft)
  selectedProviderEditKey.value = providerEditKey(draft)
  resetProviderForm(next)
}

async function saveProviderSettings() {
  persistCurrentProviderEdit()
  settingsStatus.value = 'saving'
  settingsError.value = ''
  providerTestStatus.value = 'idle'
  providerTestMessage.value = ''
  modelFetchStatus.value = 'idle'
  modelFetchMessage.value = ''
  try {
    const savingProviderKey = selectedProviderEditKey.value
    let saved = null
    if (translationProvider.value === 'llm' || hasModelProviderDraft.value) {
      saved = await saveChatModelProvider()
    }
    await invoke('save_translation_settings', {
      input: {
        provider: translationProvider.value,
        enableFallback: translationFallbackEnabled.value,
        microsoftEndpoint: microsoftForm.endpoint,
        microsoftRegion: microsoftForm.region,
        microsoftApiKey: microsoftForm.apiKey || null,
      },
    })
    await loadTranslationSettings()
    await loadModelProviders()
    if (saved) {
      const unsavedDrafts = editableProviders.value
        .filter((provider) => !provider.id && providerEditKey(provider) !== savingProviderKey)
      editableProviders.value = [
        ...modelProviders.value.map((provider) => cloneProviderForEdit(provider, `saved:${provider.id}`)),
        ...unsavedDrafts,
      ]
      selectedProviderEditKey.value = `saved:${saved.id}`
      resetProviderForm(saved)
      selectedChatModelId.value = defaultChatModelOptionId(saved)
    }
    settingsStatus.value = 'saved'
  } catch (err) {
    settingsStatus.value = 'failed'
    settingsError.value = err?.message || String(err)
  }
}

async function saveChatModelProvider() {
  return invoke('save_model_provider', {
    input: {
      id: providerForm.id || null,
      name: providerForm.name,
      providerType: providerForm.providerType,
      baseUrl: providerForm.baseUrl,
      models: providerForm.models.map((model) => ({
        key: model.key || null,
        modelId: model.modelId,
        nickname: model.nickname,
        capabilities: model.capabilities.filter((capability) => capability !== 'text'),
        enabled: model.enabled,
        isDefaultChatModel: model.key === providerForm.defaultModelKey,
      })),
      defaultModelKey: providerForm.defaultModelKey || null,
      apiKey: providerForm.apiKey || null,
      enabled: true,
      isDefault: providerForm.isDefault,
    },
  })
}

async function testProviderSettings() {
  if (translationProvider.value !== 'llm') {
    providerTestStatus.value = 'testing'
    providerTestMessage.value = ''
    try {
      const result = await invoke('test_translation_provider', {
        input: {
          provider: translationProvider.value,
          microsoftEndpoint: microsoftForm.endpoint,
          microsoftRegion: microsoftForm.region,
          microsoftApiKey: microsoftForm.apiKey || null,
        },
      })
      providerTestStatus.value = result.ok ? 'succeeded' : 'failed'
      providerTestMessage.value = result.message || ui.value.providerTestSucceeded
    } catch (err) {
      providerTestStatus.value = 'failed'
      providerTestMessage.value = err?.message || String(err)
    }
    return
  }
  providerTestStatus.value = 'testing'
  providerTestMessage.value = ''
  try {
    const result = await runChatModelProviderTest()
    providerTestStatus.value = result.ok ? 'succeeded' : 'failed'
    providerTestMessage.value = result.message || ui.value.providerTestSucceeded
  } catch (err) {
    providerTestStatus.value = 'failed'
    providerTestMessage.value = err?.message || String(err)
  }
}

async function runChatModelProviderTest() {
  return invoke('test_model_provider', {
    input: {
      id: providerForm.id || null,
      name: providerForm.name,
      providerType: providerForm.providerType,
      baseUrl: providerForm.baseUrl,
      modelId: providerForm.models.find((model) => model.key === providerForm.defaultModelKey)?.modelId
        || providerForm.models.find((model) => model.enabled)?.modelId
        || '',
      apiKey: providerForm.apiKey || null,
    },
  })
}

async function testChatModelProvider() {
  providerTestStatus.value = 'testing'
  providerTestMessage.value = ''
  try {
    const result = await runChatModelProviderTest()
    providerTestStatus.value = result.ok ? 'succeeded' : 'failed'
    providerTestMessage.value = result.message || ui.value.providerTestSucceeded
  } catch (err) {
    providerTestStatus.value = 'failed'
    providerTestMessage.value = err?.message || String(err)
  }
}

// Open a document as a tab and activate it. Appends to the working set if new,
// otherwise just activates the existing tab. The single entry point for sidebar
// selection, cross-document jumps, and restore.
function openTab(docId) {
  if (!docId) return
  if (!openTabs.value.includes(docId)) {
    openTabs.value = [...openTabs.value, docId]
  }
  if (selectedDocId.value !== docId) {
    selectedDocId.value = docId
    loadChatHistoryForDocument(docId)
    loadNotesForDocument(docId)
  }
}

// Close a tab. If it's the active one, fall to the right neighbour, else left.
// Closing the last tab is a no-op for selection — selectedDocument always renders
// some document (falling back to allDocs[0]), so we keep that doc as the sole tab
// rather than blanking selectedDocId (which would leave the reader showing a doc
// with no matching active tab). The document itself stays in the sidebar regardless.
function selectDoc(docId) {
  // Selecting a document leaves the Trending feed / knowledge graph.
  trendingView.value = false
  graphView.value = false
  const doc = allDocs.value.find((item) => item.id === docId)
  if (doc) selectedCollectionId.value = doc.collectionId ?? null
  openTab(docId)
  // Explicit selection (sidebar / tab) retargets the active conversation's focus
  // to the chosen document, so "Focus" tracks what you're reading. Citation jumps
  // go through openTab() directly (handleCitationClick) and intentionally keep the
  // current focus — you can peek at a cited paper without losing the thread.
  void handleSetSessionFocus(docId)
}

// ---- Trending Papers (online discovery) ----

// ISO-8601 week string (YYYY-Www) for HF's /papers/week scope. Computed locally
// so the backend needs no date dependency; matches huggingface.co's week ids.
function isoWeekValue(now) {
  const d = new Date(Date.UTC(now.getFullYear(), now.getMonth(), now.getDate()))
  const dayNum = (d.getUTCDay() + 6) % 7 // Mon=0 … Sun=6
  d.setUTCDate(d.getUTCDate() - dayNum + 3) // move to the week's Thursday
  const firstThursday = new Date(Date.UTC(d.getUTCFullYear(), 0, 4))
  const fd = (firstThursday.getUTCDay() + 6) % 7
  firstThursday.setUTCDate(firstThursday.getUTCDate() - fd + 3)
  const week = 1 + Math.round((d - firstThursday) / (7 * 864e5))
  return `${d.getUTCFullYear()}-W${String(week).padStart(2, '0')}`
}

function trendingScopeValue(period, now = new Date()) {
  if (period === 'weekly') return isoWeekValue(now)
  // Use UTC consistently with isoWeekValue and HF's UTC-based scopes (the daily
  // backend falls back to the latest published day if today isn't up yet).
  const y = now.getUTCFullYear()
  const m = String(now.getUTCMonth() + 1).padStart(2, '0')
  if (period === 'monthly') return `${y}-${m}`
  return `${y}-${m}-${String(now.getUTCDate()).padStart(2, '0')}`
}

// Monotonic token so an out-of-order fetch response (fast tab switching) can't
// overwrite the list for the period the user is actually viewing.
let trendingFetchToken = 0

async function fetchTrending() {
  const token = ++trendingFetchToken
  const period = trendingPeriod.value
  trendingStatus.value = 'loading'
  trendingError.value = ''
  try {
    const papers = await invoke('fetch_trending_papers', {
      period,
      value: trendingScopeValue(period),
      limit: 30,
    })
    if (token !== trendingFetchToken) return // a newer fetch superseded this one
    trendingPapers.value = Array.isArray(papers) ? papers : []
    trendingStatus.value = 'loaded'
  } catch (err) {
    if (token !== trendingFetchToken) return
    trendingError.value = err?.message || String(err)
    trendingStatus.value = 'failed'
  }
}

function handleTrendingPeriod(period) {
  if (!['daily', 'weekly', 'monthly'].includes(period) || period === trendingPeriod.value) return
  trendingPeriod.value = period
  fetchTrending()
}

async function fetchKnowledgeGraph() {
  graphStatus.value = 'loading'
  graphError.value = ''
  try {
    // Collapse alias variants across the library once per session (normalize_key
    // is deterministic, so re-running it on every open is a wasted full-table
    // pass). New artifacts are already normalized at write time.
    if (!knowledgeConsolidated) {
      knowledgeConsolidated = true
      await invoke('consolidate_knowledge').catch(() => {})
    }
    graphData.value = await invoke('get_knowledge_graph')
    graphStatus.value = 'loaded'
  } catch (err) {
    graphError.value = err?.message || String(err)
    graphStatus.value = 'failed'
  }
}

function handleOpenGraph() {
  // Clicking the sidebar entry again while the graph is open toggles it closed.
  if (graphView.value) {
    handleCloseGraph()
    return
  }
  graphFocusOnOpen.value = false
  openGraphView()
}

// Return to the "ask my knowledge base" home: leave any doc / trending / graph.
function goHome() {
  trendingView.value = false
  graphView.value = false
  selectedDocId.value = ''
}

// Exit the global graph back to the reader (close button / ESC / sidebar toggle).
function handleCloseGraph() {
  graphView.value = false
}

// "View in graph" from the knowledge card: same view, but pre-focused on the
// document being read (one click instead of sidebar → graph → Focus current).
function handleOpenGraphFocused() {
  graphFocusOnOpen.value = true
  openGraphView()
}

function openGraphView() {
  graphView.value = true
  trendingView.value = false
  if (graphStatus.value === 'idle' || graphStatus.value === 'failed') {
    fetchKnowledgeGraph()
  }
}

// The palette's action list. Built here rather than inside the component so the
// component stays a renderer — and so an action that is switched off (trending,
// knowledge graph) simply isn't offered instead of failing silently when run.
const paletteActions = computed(() => {
  const list = [
    { id: 'home', label: ui.value.commandActionHome, hint: '' },
    { id: 'new-note', label: ui.value.commandActionNewNote, hint: '' },
  ]
  if (knowledgeEnabled.value) {
    list.push({ id: 'graph', label: ui.value.commandActionGraph, hint: '' })
  }
  if (trendingEnabled.value) {
    list.push({ id: 'trending', label: ui.value.commandActionTrending, hint: '' })
  }
  list.push({ id: 'settings', label: ui.value.commandActionSettings, hint: '' })
  return list
})

function runPaletteAction(id) {
  if (id === 'home') goHome()
  else if (id === 'new-note') handleCreateNote()
  else if (id === 'graph') openGraphView()
  else if (id === 'trending') handleOpenTrending()
  else if (id === 'settings') openSettings()
}

function handleOpenTrending() {
  if (!trendingEnabled.value) return
  graphView.value = false
  trendingView.value = true
  if (trendingStatus.value === 'idle' || trendingStatus.value === 'failed') {
    fetchTrending()
  }
}

async function handleAddTrendingPaper(paper) {
  const arxivId = String(paper?.arxivId || '').trim()
  if (!arxivId || trendingAddingIds.value.includes(arxivId)) return
  trendingAddError.value = ''
  trendingAddingIds.value = [...trendingAddingIds.value, arxivId]
  try {
    const result = await invoke('add_trending_paper', {
      arxivId,
      title: paper?.title || null,
    })
    if (result?.snapshot) upsertWorkspaceRootSnapshot(result.snapshot)
    if (result?.documentId) selectDoc(result.documentId) // opens it; leaves the feed
  } catch (err) {
    // Surfaced in the discovery view (where the user is), not just the sidebar.
    trendingAddError.value = err?.message || String(err)
  } finally {
    trendingAddingIds.value = trendingAddingIds.value.filter((id) => id !== arxivId)
  }
}

function handleOpenHf(url) {
  if (!url) return
  invoke('open_external_url', { url }).catch((err) => {
    trendingAddError.value = err?.message || String(err)
  })
}

function openExternalLink(url) {
  if (url) invoke('open_external_url', { url }).catch(() => {})
}

// ---- Update notification (notify-only) ----

async function checkForUpdate() {
  try {
    const info = await invoke('check_for_update')
    if (info?.hasUpdate) updateInfo.value = info
  } catch {
    // Offline / rate-limited / network error: no banner, never blocks startup.
  }
}

function openUpdatePage() {
  const url = updateInfo.value?.url
  if (url) invoke('open_external_url', { url }).catch(() => {})
}

function dismissUpdateBanner() {
  if (updateInfo.value?.latest) dismissedUpdateVersion.value = updateInfo.value.latest
}

// Fetch when the discovery feed BECOMES visible (explicit open, or the user
// closes their last document). NOT { immediate } — that fired on every cold start
// (the workspace loads async, so the reader is briefly empty), leaking a network
// call for users who never use the feature. The genuinely-empty-workspace startup
// case is handled after the workspace finishes loading (loadLastWorkspace).
watch(showTrending, (visible) => {
  if (visible && trendingStatus.value === 'idle') fetchTrending()
})

// Any path that selects a document leaves the discovery feed (drag-drop import,
// the "+" picker, add-folder/scan — not just selectDoc).
watch(selectedDocId, (documentId) => {
  trendingView.value = false
  graphView.value = false
  // Show the open document's knowledge card; ensure it gets precipitated.
  loadKnowledgeCard(documentId)
  const doc = allDocs.value.find((item) => item.id === documentId)
  if (doc) scheduleDocumentPrecipitation(doc)
})

// Turning the feature on retroactively precipitates the existing library.
watch(knowledgeEnabled, (enabled) => {
  if (enabled) backfillPrecipitation()
})

function handleCitationClick(citation) {
  // Discovery-tool results (list_trending_papers / search_library_knowledge) are
  // reference chips, not PDF anchors: page 0, no block. Never jump the reader to
  // page 0 or paint a bogus highlight — just open the referenced library doc if
  // there is one (trending has no documentId, so this is a no-op for it).
  const anchored = Number(citation.page) > 0 || Boolean(citation.blockId)
  if (!anchored) {
    if (citation.documentId && citation.documentId !== selectedDocId.value) {
      openTab(citation.documentId)
    }
    return
  }
  // A citation may belong to an @-referenced document the user isn't reading. In
  // that case switch to (or open) its tab first; the actual page/highlight is
  // applied after watch(selectedDocument) resets reader state (see pendingCitationJump).
  if (citation.documentId && citation.documentId !== selectedDocId.value) {
    pendingCitationJump = citation
    openTab(citation.documentId)
    return
  }
  applyCitationJump(citation)
}

// Move the reader to a citation's page and paint its highlight. Shared by direct
// (same-document) clicks and the deferred cross-document jump.
function applyCitationJump(citation) {
  activePage.value = citation.page
  activeBlockId.value = citation.blockId
  activeCitationId.value = citation.id
  activeHighlight.value = createHighlight(citation)
  // Office sources have no PDF geometry (page 0, empty bbox), so the reader's
  // page/highlight props mean nothing to them. Hand the whole citation to the
  // Office viewer instead — it anchors on the quote's text (docx) or its
  // `Sheet!row` reference (xlsx).
  activeCitation.value = citation
}

function nextLocalId(prefix) {
  localMessageCounter += 1
  return `${prefix}-${Date.now()}-${localMessageCounter}`
}

// Knowledge-base pivot (P2): create a blank standalone note, merge it into the
// workspace tree (virtual Knowledge Base root) and open it in the Markdown
// editor. The backend already enqueues its first index pass.
async function handleCreateNote(title = '') {
  // `title` is a plain string when invoked from a [[wikilink]] that doesn't
  // resolve yet (create-on-click); the new-note buttons pass no argument.
  const noteTitle = typeof title === 'string' ? title.trim() : ''
  try {
    const result = await invoke('create_note_source', {
      input: { title: noteTitle, bodyMd: '', collectionId: selectedCollectionId.value ?? null },
    })
    if (result?.snapshot) {
      const { root } = upsertWorkspaceRootSnapshot(result.snapshot)
      // Make sure the new note is visible (un-collapse the knowledge root).
      if (root) root.collapsed = false
    }
    if (result?.documentId) {
      await nextTick()
      selectDoc(result.documentId)
    }
  } catch (err) {
    workspaceError.value = err?.message || String(err)
  }
}

// Knowledge-base pivot (P2): clip a web page into the knowledge base. The
// backend fetches + strips the page, stores it as an editable 'web' source and
// indexes it; we open the captured Markdown in the editor.
async function handleClipWebPage(url) {
  const target = String(url || '').trim()
  if (!target || clipBusy.value) return
  clipBusy.value = true
  workspaceError.value = ''
  try {
    const result = await invoke('clip_web_page', { input: { url: target, collectionId: selectedCollectionId.value ?? null } })
    if (result?.snapshot) {
      const { root } = upsertWorkspaceRootSnapshot(result.snapshot)
      if (root) root.collapsed = false
    }
    if (result?.documentId) {
      await nextTick()
      selectDoc(result.documentId)
    }
  } catch (err) {
    workspaceError.value = err?.message || String(err)
  } finally {
    clipBusy.value = false
  }
}

// Knowledge-base pivot: load the flat collection list (sorted by position/name).
async function loadCollections() {
  try {
    collections.value = await invoke('load_collections')
  } catch (err) {
    console.warn('load_collections failed', err)
  }
}

async function handleCreateCollection(parentId = null) {
  const name = ui.value.newCollectionName || 'New collection'
  try {
    const created = await invoke('create_collection', {
      input: { name, parentId: parentId ?? null },
    })
    await loadCollections()
    if (created?.id) selectedCollectionId.value = created.id
  } catch (err) {
    workspaceError.value = err?.message || String(err)
  }
}

async function handleRenameCollection({ id, name } = {}) {
  if (!id) return
  const nextName = String(name || '').trim()
  if (!nextName) return
  try {
    await invoke('rename_collection', { input: { id, name: nextName } })
    await loadCollections()
  } catch (err) {
    workspaceError.value = err?.message || String(err)
  }
}

async function handleDeleteCollection(id) {
  if (!id) return
  try {
    await invoke('delete_collection', { id })
    // The backend unfiled the whole deleted subtree (collection_id → null).
    // Refresh the collection list, then reconcile local docs in place — cheaper
    // and less disruptive than a full workspace reload (which would reset the
    // open tab / selection / chat). Any doc still pointing at a now-gone
    // collection drops to unfiled.
    await loadCollections()
    const known = new Set(collections.value.map((collection) => collection.id))
    for (const doc of allDocs.value) {
      if (doc.collectionId && !known.has(doc.collectionId)) doc.collectionId = null
    }
    if (selectedCollectionId.value === id) selectedCollectionId.value = null
  } catch (err) {
    workspaceError.value = err?.message || String(err)
  }
}

function handleSelectCollection(id) {
  selectedCollectionId.value = id
}

// Empty the Unfiled inbox: delete every source not in a (still-existing)
// collection. Non-destructive to the files on disk — delete_document only drops
// the knowledge-base record. Reuses the single-doc delete path per source.
async function handleClearUnfiled() {
  // The backend is the single source of truth for "unfiled" (collection_id IS
  // NULL) and deletes in one transaction. This deliberately does NOT filter on
  // the frontend: a stale/half-loaded collections list must never be able to
  // classify filed documents as unfiled and wipe the library. Chat history is
  // preserved (chat_turns.document_id is ON DELETE SET NULL).
  workspaceError.value = ''
  let deletedIds = []
  try {
    deletedIds = await invoke('clear_unfiled_documents')
  } catch (err) {
    workspaceError.value = err?.message || String(err)
    return
  }
  if (!Array.isArray(deletedIds) || !deletedIds.length) return
  for (const id of deletedIds) {
    removeDocFromWorkspaceState(id)
    openTabs.value = openTabs.value.filter((tabId) => tabId !== id)
    precipitationQueued.delete(id)
    visualIndexRuns.delete(id)
  }
  if (!allDocs.value.some((doc) => doc.id === selectedDocId.value)) {
    selectedDocId.value = ''
  }
}

// The user accepted an agent-proposed rewrite. It is applied through the open editor
// (the single writer) rather than written to the database, so it can never race the
// autosave. Requires the proposal's note to be the one on screen: applying to a
// document the user isn't looking at would be an invisible edit.
/* NoteEditor is a lazily-loaded async component, so after selecting the note its
   chunk still has to download, mount, and register this ref. A single nextTick()
   never wins that race from a cold start — which is exactly the case where the
   user reaches for Apply, having just reopened the app with no note open — so it
   reported "open the note" for a note it had just opened. Poll by frame instead,
   with a ceiling so a document that will never mount an editor still fails. */
const NOTE_EDITOR_WAIT_MS = 5000

async function waitForNoteEditor() {
  const deadline = Date.now() + NOTE_EDITOR_WAIT_MS
  while (Date.now() < deadline) {
    if (noteEditorRef.value?.applyProposal) return noteEditorRef.value
    await new Promise((resolve) => requestAnimationFrame(resolve))
  }
  return null
}

async function handleApplyNoteEdit(proposal = {}) {
  const { documentId, onResult } = proposal
  const target = documentId ? allDocs.value.find((doc) => doc.id === documentId) : selectedDocument.value
  // Bail before the wait when the target cannot host an editor at all — spending
  // five seconds to reach the same conclusion helps nobody.
  if (documentId && target && !isEditableSourceDoc(target)) {
    workspaceError.value = ui.value.proposedEditNeedsNote || 'Open the note to apply this edit.'
    onResult?.(false)
    return
  }
  if (documentId && documentId !== selectedDocId.value) {
    selectDoc(documentId)
    await nextTick()
  }
  const editor = await waitForNoteEditor()
  if (!editor) {
    workspaceError.value = ui.value.proposedEditNeedsNote || 'Open the note to apply this edit.'
    onResult?.(false)
    return
  }
  try {
    await editor.applyProposal(proposal)
    workspaceError.value = ''
    onResult?.(true)
  } catch (err) {
    // A precise edit refused to apply because the note moved on under it. Say so
    // rather than falling back to a whole-note overwrite, which is exactly the
    // silent data loss precise edits exist to prevent.
    workspaceError.value = err?.message || String(err)
    onResult?.(false)
  }
}

// Show a source's file in Finder / Explorer. Imports reference files in place
// rather than copying them, so this is how the user answers "where is it?" — and
// it reports a file that was moved or deleted outside the app instead of failing
// silently.
async function handleRevealDoc(doc) {
  if (!doc?.id) return
  try {
    await invoke('reveal_document_in_file_manager', { documentId: doc.id })
  } catch (err) {
    workspaceError.value = err?.message || String(err)
  }
}

async function handleMoveDocToCollection({ docId, collectionId } = {}) {
  if (!docId) return
  const target = collectionId ?? null
  try {
    await invoke('move_document_to_collection', {
      input: { documentId: docId, collectionId: target },
    })
    const doc = allDocs.value.find((item) => item.id === docId)
    if (doc) doc.collectionId = target
  } catch (err) {
    workspaceError.value = err?.message || String(err)
  }
}

// C-d: nest / reparent a collection by dragging it onto another (or to top level
// when dropped on Unfiled). The backend rejects cycles.
async function handleMoveCollection({ id, parentId } = {}) {
  if (!id || id === parentId) return
  try {
    await invoke('move_collection', { input: { id, parentId: parentId ?? null } })
    await loadCollections()
  } catch (err) {
    workspaceError.value = err?.message || String(err)
  }
}

// Manual drag-reorder: the sidebar sends the full sibling id list in its new
// order. Apply it locally first (write each row's position = its index so the
// tree re-sorts instantly), then persist. `parentId`/`collectionId` is the
// sibling scope (null = the tree root).
async function handleReorderDocuments({ collectionId = null, orderedIds } = {}) {
  if (!Array.isArray(orderedIds) || !orderedIds.length) return
  const scope = collectionId ?? null
  const rank = new Map(orderedIds.map((id, index) => [id, index]))
  for (const doc of allDocs.value) {
    if ((doc.collectionId ?? null) === scope && rank.has(doc.id)) {
      doc.position = rank.get(doc.id)
    }
  }
  try {
    await invoke('reorder_documents', { input: { parentId: scope, orderedIds } })
  } catch (err) {
    workspaceError.value = err?.message || String(err)
  }
}

async function handleReorderCollections({ parentId = null, orderedIds } = {}) {
  if (!Array.isArray(orderedIds) || !orderedIds.length) return
  const scope = parentId ?? null
  const rank = new Map(orderedIds.map((id, index) => [id, index]))
  for (const collection of collections.value) {
    if ((collection.parentId ?? null) === scope && rank.has(collection.id)) {
      collection.position = rank.get(collection.id)
    }
  }
  try {
    await invoke('reorder_collections', { input: { parentId: scope, orderedIds } })
  } catch (err) {
    workspaceError.value = err?.message || String(err)
  }
}

// After a note saves, reflect the new title in the tree. Index status is
// deliberately NOT written here: the reindex already emits queued → indexing →
// indexed, and those events drive the status in the right order. This callback runs
// after `await invoke('update_note_source')` resolves, which for a short note is
// *after* the whole index has already finished — so optimistically setting
// 'indexing' here clobbered the final 'indexed' with no event left to correct it,
// pinning the UI on "Indexing…" forever. Autosave made that race fire on every edit.
function handleNoteSaved(payload) {
  const target = allDocs.value.find((doc) => doc.id === payload?.documentId)
  if (!target) return
  if (payload.title) {
    target.title = payload.title
    target.shortTitle = payload.title
  }
}

async function handleSend(payload, selection = null) {
  // Flush the note editor first. The agent reads a note's Markdown from the database
  // (read_note_source), and autosave is debounced — so asking "tidy this up" right
  // after typing would otherwise hand the agent a version missing the last sentence.
  // No-op unless an editable source is open with unsaved edits.
  await noteEditorRef.value?.flushSave?.()
  const session = await ensureActiveSession()
  if (!session) return
  // The focus document is the session's default retrieval target; readiness and
  // citations are scoped to it. KB pivot (P1): with no focus document the chat
  // goes library-wide ("ask my knowledge base") — focusDocId is empty and the
  // backend seeds retrieval across the whole library.
  const doc = activeFocusDoc.value
  if (!doc) return
  const focusDocId = doc.id && doc.id !== 'empty' ? doc.id : ''
  const libraryWide = !focusDocId
  if (!session.chatHistoryLoaded) {
    await ensureSessionHistory(session)
  }
  const payloadObject = typeof payload === 'string' ? null : payload
  const messageText = typeof payload === 'string' ? payload : String(payload?.text || '')
  const imageDataUrl = typeof payload === 'string' ? '' : String(payload?.imageDataUrl || '')
  const selectedQuote = selection || (payloadObject?.ignoreSelection ? null : lastSelection.value)
  const maxRetrievalSteps = Number(payloadObject?.maxRetrievalSteps || 20)
  const retrievalAttemptOffset = Number(payloadObject?.retrievalAttemptOffset || 0)
  // "@-referenced" papers selected in the composer; never includes the active doc.
  const referenceDocumentIds = Array.isArray(payloadObject?.mentionedDocIds)
    ? payloadObject.mentionedDocIds.filter((id) => id && id !== focusDocId)
    : []
  if (!chatModelConfigured.value) return
  const { providerId, modelKey } = parseChatModelOptionId(selectedChatModelId.value)
  if (!messageText.trim() && !imageDataUrl) return
  let citations = selectedQuote
    ? [{
      id: nextLocalId('selection-c'),
      label: '[1]',
      page: selectedQuote.page,
      blockId: selectedQuote.blockId || '',
      quote: selectedQuote.text,
      bboxList: selectedQuote.bboxList || [],
      documentId: focusDocId,
      source: 'selection',
      sourceType: selectedQuote.sourceType || 'selection',
    }]
    : []

  const userMessageId = nextLocalId('u')
  const assistantMessageId = nextLocalId('a')
  const activityEventId = `agent-${assistantMessageId}`
  chatStreamDebug('send', {
    documentId: focusDocId,
    eventId: activityEventId,
    modelProviderId: providerId,
    modelKey,
    questionLength: messageText.trim().length,
    hasImage: Boolean(imageDataUrl),
  })
  session.messages.push({
    id: userMessageId,
    sessionId: session.id,
    role: 'user',
    content: {
      en: messageText,
      zh: messageText,
    },
    citations: [],
    imageDataUrl: imageDataUrl || null,
    mentionedDocumentIds: referenceDocumentIds,
  })
  // Still indexing (or index went stale): the backend has no evidence to answer
  // from yet, so reply in natural language instead of querying. A selected quote
  // can be answered from the selection, so that path proceeds normally. Library-
  // wide chat has no single document to be "ready", so it always proceeds.
  if (!libraryWide && !doc.chatReady && !selectedQuote) {
    session.messages.push({
      id: assistantMessageId,
      sessionId: session.id,
      role: 'assistant',
      content: { en: ui.value.docStillIndexing, zh: ui.value.docStillIndexing },
      citations: [],
      status: 'succeeded',
      activityEvents: [],
      reasoningContent: '',
      originalQuestion: messageText.trim() || ui.value.imageOnlyPrompt,
    })
    session.updatedAt = Math.floor(Date.now() / 1000)
    return
  }
  session.messages.push({
    id: assistantMessageId,
    sessionId: session.id,
    role: 'assistant',
    content: {
      en: '',
      zh: '',
    },
    citations,
    status: 'running',
    activityEventId,
    activityEvents: [],
    reasoningContent: '',
    originalQuestion: messageText.trim() || ui.value.imageOnlyPrompt,
    maxRetrievalSteps,
    retrievalAttemptOffset,
  })
  session.updatedAt = Math.floor(Date.now() / 1000)
  if (citations.length && selectedDocument.value.id === doc.id) {
    const firstCitation = citations[0]
    activeCitationId.value = firstCitation.id
    activeHighlight.value = firstCitation.source === 'selection' ? null : createHighlight(firstCitation)
  }
  if (selectedQuote && lastSelection.value?.text === selectedQuote.text) {
    lastSelection.value = null
  }

  try {
    await invoke('ask_document_stream', {
      input: {
        documentId: focusDocId,
        sessionId: session.id,
        question: messageText.trim() || ui.value.imageOnlyPrompt,
        locale: locale.value,
        modelProviderId: chatModelConfigured.value ? providerId : null,
        modelKey: chatModelConfigured.value ? modelKey : null,
        selectedText: selectedQuote?.text || '',
        selectedBlockId: selectedQuote?.blockId || '',
        selectedBboxList: selectedQuote?.bboxList || [],
        imageDataUrl: imageDataUrl || null,
        page: selectedQuote?.page || citations[0]?.page || activePage.value || null,
        viewportContext: {
          activePage: activePage.value || null,
          visiblePages: activePage.value ? [activePage.value] : [],
          selectionPreview: selectedQuote?.text ? selectedQuote.text.slice(0, 500) : '',
          capturedAt: Date.now(),
          sensitivity: 'normal',
          source: 'pdf-viewer',
        },
        referenceDocumentIds,
        maxRetrievalSteps,
        retrievalAttemptOffset,
        activityEventId,
        knowledgeEnabled: knowledgeEnabled.value,
        webEnabled: webSearchEnabled.value,
        // Ambient surface so the agent can resolve "the current trending papers"
        // and knows when the question isn't about the focus PDF.
        viewContext: {
          surface: showTrending.value ? 'trending' : (showGraph.value ? 'graph' : 'reader'),
          trendingPeriod: trendingPeriod.value,
        },
      },
    })
  } catch (err) {
    const assistantMessage = session.messages.find((message) => message.id === assistantMessageId)
    if (!assistantMessage) return
    const error = err?.message || String(err)
    assistantMessage.content = {
      en: `${messages.en.chatFailed}: ${error}`,
      zh: `${messages.zh.chatFailed}: ${error}`,
    }
    assistantMessage.status = 'failed'
  }
}

function findMessageByActivityEventId(eventId) {
  for (const session of chatSessions.values()) {
    const found = (session.messages || []).find((item) => item.activityEventId === eventId)
    if (found) return found
  }
  return null
}

function messageDisplayText(message) {
  if (!message) return ''
  return typeof message.content === 'object'
    ? (message.content[locale.value] || message.content.en || message.content.zh || '')
    : String(message.content || '')
}

// Render the active chat session as Markdown: per-turn user question + assistant
// answer + the answer's sources (page/section/quote). Reasoning and trace are
// intentionally left out for a clean, shareable document.
function buildChatMarkdown(session) {
  const title = sessionTabTitle(session) || ui.value.newSession
  const dateStr = new Date().toISOString().slice(0, 10)
  const context = session.focusDocTitle || chatBrowsingLabel.value || ''
  const userLabel = locale.value === 'zh' ? '用户' : 'User'
  const out = [`# ${title}`, '', `> Lumenfolio · ${dateStr}${context ? ` · ${context}` : ''}`, '']
  for (const message of session.messages || []) {
    const text = messageDisplayText(message).trim()
    if (message.role === 'user') {
      out.push(`## 🧑 ${userLabel}`, '')
      out.push(text || '_(empty)_')
      if (message.imageDataUrl) out.push('', `_[${locale.value === 'zh' ? '附图' : 'image attached'}]_`)
      out.push('')
    } else if (message.role === 'assistant') {
      const provider = message.provider ? ` · ${message.provider}` : ''
      out.push(`## 🤖 ${locale.value === 'zh' ? '助手' : 'Assistant'}${provider}`, '')
      out.push(text || '_(no answer)_')
      const citations = message.citations || []
      if (citations.length) {
        out.push('', `**${ui.value.sources}:**`)
        for (const c of citations) {
          const page = Number(c.page) > 0 ? ` p${c.page}` : ''
          const section = c.sectionTitle ? ` — ${c.sectionTitle}` : ''
          const quote = String(c.quote || '').replace(/\s+/g, ' ').trim()
          const quoteShort = quote.length > 200 ? `${quote.slice(0, 200)}…` : quote
          out.push(`- ${c.label || '•'}${page}${section}${quoteShort ? `: ${quoteShort}` : ''}`)
        }
      }
      out.push('')
    }
    out.push('---', '')
  }
  return out.join('\n')
}

const transientToast = ref('')
let transientToastTimer = null
function showToast(text) {
  transientToast.value = text
  if (transientToastTimer) window.clearTimeout(transientToastTimer)
  transientToastTimer = window.setTimeout(() => { transientToast.value = '' }, 2600)
}

async function handleExportChat() {
  const session = activeSession.value
  if (!session || !(session.messages || []).some((message) => messageDisplayText(message).trim())) {
    showToast(ui.value.exportChatEmpty)
    return
  }
  const markdown = buildChatMarkdown(session)
  const safeTitle = ((sessionTabTitle(session) || 'chat')
    .replace(/[\\/:*?"<>|]+/g, '-')
    .slice(0, 60)
    .trim()) || 'chat'
  const defaultName = `lumenfolio-${safeTitle}-${new Date().toISOString().slice(0, 10)}.md`
  try {
    const saved = await invoke('export_markdown_file', { defaultName, content: markdown })
    if (saved) showToast(ui.value.exportChatDone)
  } catch (err) {
    console.warn('Failed to export chat', err)
  }
}

function setMessageDisplayText(message, text) {
  message.content = {
    en: text,
    zh: text,
  }
}

function assistantStreamState(eventId) {
  if (!assistantStreamTargets.has(eventId)) {
    assistantStreamTargets.set(eventId, {
      target: '',
      doneResult: null,
      finalReplacement: '',
    })
  }
  return assistantStreamTargets.get(eventId)
}

function queueAssistantStreamText(eventId, text) {
  if (!eventId || !text || clearedActivityEventIds.has(eventId)) return
  const message = findMessageByActivityEventId(eventId)
  if (!message) return
  const state = assistantStreamState(eventId)
  state.target += text
  chatStreamDebug('delta queued', {
    eventId,
    deltaLength: text.length,
    targetLength: state.target.length,
    visibleLength: messageDisplayText(message).length,
    preview: text.slice(0, 24),
  })
  scheduleAssistantStreamDrain()
}

function markAssistantStreamDone(eventId, result) {
  if (!eventId || clearedActivityEventIds.has(eventId)) return
  const message = findMessageByActivityEventId(eventId)
  if (!message) return
  const state = assistantStreamState(eventId)
  const finalAnswer = String(result?.answer || '')
  if (finalAnswer) {
    if (!state.target) {
      state.target = finalAnswer
    } else if (finalAnswer.startsWith(state.target)) {
      state.target = finalAnswer
    } else if (state.target !== finalAnswer) {
      state.finalReplacement = finalAnswer
    }
  }
  state.doneResult = result || {}
  chatStreamDebug('done queued', {
    eventId,
    finalAnswerLength: finalAnswer.length,
    targetLength: state.target.length,
    finalReplacementLength: state.finalReplacement.length,
    visibleLength: messageDisplayText(message).length,
  })
  scheduleAssistantStreamDrain()
}

function scheduleAssistantStreamDrain() {
  if (assistantStreamDrainTimer) return
  assistantStreamDrainTimer = window.setTimeout(() => {
    assistantStreamDrainTimer = null
    drainAssistantStreamTargets()
  }, ASSISTANT_STREAM_DRAIN_MS)
}

function drainAssistantStreamTargets() {
  let hasPendingText = false
  for (const [eventId, state] of assistantStreamTargets.entries()) {
    const message = findMessageByActivityEventId(eventId)
    if (!message) {
      if (state.doneResult) assistantStreamTargets.delete(eventId)
      continue
    }
    const currentText = messageDisplayText(message)
    if (!state.target.startsWith(currentText)) {
      setMessageDisplayText(message, '')
    }
    const refreshedText = messageDisplayText(message)
    if (state.target.length > refreshedText.length) {
      const remaining = state.target.length - refreshedText.length
      const increment = Math.max(
        ASSISTANT_STREAM_CHARS_PER_TICK,
        Math.ceil(remaining / ASSISTANT_STREAM_CATCHUP_DIVISOR),
      )
      const nextText = state.target.slice(0, refreshedText.length + increment)
      setMessageDisplayText(message, nextText)
      if (refreshedText.length === 0 || nextText.length === state.target.length || nextText.length % 20 === 0) {
        chatStreamDebug('drain tick', {
          eventId,
          visibleLength: nextText.length,
          targetLength: state.target.length,
          doneQueued: Boolean(state.doneResult),
        })
      }
      hasPendingText = true
      continue
    }
    if (state.doneResult) {
      if (state.finalReplacement) {
        setMessageDisplayText(message, state.finalReplacement)
      }
      chatStreamDebug('drain complete', {
        eventId,
        finalVisibleLength: messageDisplayText(message).length,
        finalReplacementLength: state.finalReplacement.length,
      })
      applyAskDocumentMetadata(message, state.doneResult)
      assistantStreamTargets.delete(eventId)
    }
  }
  if (hasPendingText || Array.from(assistantStreamTargets.values()).some((state) => state.doneResult)) {
    scheduleAssistantStreamDrain()
  }
}

function clearAssistantStreamState(eventId) {
  assistantStreamTargets.delete(eventId)
}

function applyAskDocumentMetadata(message, result) {
  message.claims = result?.claims || []
  message.provider = result?.provider || ''
  // Trust the final (server-cleaned) reasoning over the streamed accumulation, so
  // leaked tool-call markup stripped on the backend doesn't linger in the drawer.
  message.reasoningContent = result?.reasoningContent || ''
  message.citations = result?.citations?.length ? result.citations : message.citations || []
  message.retrievalTrace = result?.retrievalTrace || null
  message.activityEvents = result?.retrievalTrace?.events || message.activityEvents || []
  message.canContinueRetrieval = false
  message.continuationRequest = null
  message.status = 'succeeded'
  // First successful answer in a session: condense it into a tab title.
  if (message.sessionId) {
    const session = chatSessions.get(message.sessionId)
    if (session) void maybeGenerateSessionTitle(session)
  }
  // After the answer settles, surface related papers + prior knowledge for it.
  void loadTurnRecommendations(message)
}

// Reloaded history turns also carry recommendations (the backend still has their
// claims). Fetch only the most recent assistant turns so opening a long session
// doesn't fire dozens of queries; fire-and-forget so it never blocks the load.
function hydrateHistoryRecommendations(session) {
  if (!knowledgeEnabled.value || !session) return
  const turns = (session.messages || []).filter(
    (message) => message.role === 'assistant'
      && (message.turnId || message.activityEventId)
      && !message.recommendations,
  )
  for (const message of turns.slice(-10)) {
    void loadTurnRecommendations(message)
  }
}

// Post-answer intelligent recommendations: ask the backend for other library
// papers + precipitated claims relevant to *this* turn (graph-only, no LLM). The
// turn's persisted id is its activityEventId (== backend turn_id). Best-effort —
// recommendations are a nicety, never block or surface an error.
async function loadTurnRecommendations(message) {
  if (!knowledgeEnabled.value || !message) return
  const turnId = String(message.turnId || message.activityEventId || '').trim()
  if (!turnId) return
  try {
    const payload = await invoke('get_turn_recommendations', { turnId })
    const related = Array.isArray(payload?.related) ? payload.related : []
    const claims = Array.isArray(payload?.claims) ? payload.claims : []
    if (!related.length && !claims.length) return
    message.recommendations = {
      related,
      claims,
      confident: Boolean(payload?.confident),
    }
  } catch {
    // Ignore: a missing recommendation block is not worth a banner.
  }
}

// Stop button: ask the backend to cancel the in-flight generation. The HTTP path
// returns its partial via the normal done event; non-streaming paths emit a
// "stopped" event we finalize below. Optimistically nothing else is needed — the
// backend cancels within ~150ms and finalizes the message.
async function handleStopGeneration(eventId) {
  if (!eventId) return
  // Paint whatever we have in full and clear "running" right now. The backend
  // cancel races in parallel, but it may already have finished (the typewriter
  // was just catching up), so a local flush guarantees the click does something
  // visible instead of waiting out the reveal.
  flushAssistantStream(eventId)
  try {
    await invoke('stop_ask_document', { eventId })
  } catch (err) {
    console.warn('Failed to stop generation', err)
  }
}

// Reveal the full accumulated target immediately and finalize the message. Used
// by stop: turns the gradual reveal into an instant one so the bubble settles
// and the running state (and stop button) clear without delay.
function flushAssistantStream(eventId) {
  if (!eventId) return
  const message = findMessageByActivityEventId(eventId)
  if (!message || message.status !== 'running') return
  const state = assistantStreamState(eventId)
  const finalText = state.finalReplacement || state.target || messageDisplayText(message)
  if (state.doneResult) {
    setMessageDisplayText(message, finalText)
    applyAskDocumentMetadata(message, state.doneResult)
  } else {
    // Backend hasn't reported done yet (still mid-generation): keep the partial,
    // or a short note if nothing streamed, and optimistically settle the bubble.
    setMessageDisplayText(message, finalText || ui.value.generationStopped)
    message.status = 'succeeded'
  }
  assistantStreamTargets.delete(eventId)
}

// A non-streaming generation (agent loop / local CLI) that the user stopped:
// finalize the message with whatever streamed so far, no error banner.
function applyAskDocumentStopped(eventId) {
  if (!eventId || clearedActivityEventIds.has(eventId)) return
  const message = findMessageByActivityEventId(eventId)
  if (!message) return
  // Keep whatever streamed; if nothing did (non-streaming path stopped early),
  // show a short note rather than an empty bubble.
  const partial = assistantStreamState(eventId).target || messageDisplayText(message)
  markAssistantStreamDone(eventId, { answer: partial || ui.value.generationStopped })
}

function applyAskDocumentResult(eventId, result) {
  chatStreamDebug('done event received', {
    eventId,
    answerLength: String(result?.answer || '').length,
    hasMessage: Boolean(findMessageByActivityEventId(eventId)),
  })
  if (clearedActivityEventIds.has(eventId)) {
    const sessionId = clearedActivityEventIds.get(eventId)
    clearAssistantStreamState(eventId)
    clearedActivityEventIds.delete(eventId)
    invoke('clear_chat_turns', { input: { sessionId, turnIds: [eventId] } }).catch((err) => {
      console.warn(ui.value.clearChatHistoryFailed, err)
    })
    return
  }
  const message = findMessageByActivityEventId(eventId)
  if (!message) return
  markAssistantStreamDone(eventId, result)
}

// Chat history is now owned by the active session, not the document. The many
// existing call sites (doc select, post-reindex refresh) funnel through here;
// we simply ensure the active session's turns are loaded. The docId argument is
// retained for call-site compatibility but no longer scopes the load.
async function loadChatHistoryForDocument(_docId) {
  const session = activeSession.value
  if (session) return ensureSessionHistory(session)
}

async function loadNotesForDocument(docId) {
  const doc = allDocs.value.find((item) => item.id === docId)
  if (!doc || doc.id === 'empty' || doc.notesLoaded) return
  if (doc.notesLoading) return doc.notesLoading
  doc.notesLoading = (async () => {
    try {
      const notes = await invoke('load_notes', { input: { documentId: doc.id } })
      doc.notes = Array.isArray(notes) ? notes : []
      doc.notesLoaded = true
    } catch (err) {
      doc.notesLoaded = false
      console.warn('Failed to load notes', err)
    } finally {
      doc.notesLoading = null
    }
  })()
  return doc.notesLoading
}

function setRightPaneTab(tab) {
  notesDrawerOpen.value = tab === 'notes'
  if (notesDrawerOpen.value && rightCollapsed.value) rightCollapsed.value = false
}

function toggleNotesDrawer() {
  setRightPaneTab(notesDrawerOpen.value ? 'chat' : 'notes')
}

function openNoteComposer(selection) {
  const selected = selection || lastSelection.value
  if (!selected) return
  handleSelection(selected)
  noteComposer.value = {
    open: true,
    mode: 'create',
    quoteText: selected.text || '',
    content: '',
    noteId: '',
    selection: selected,
  }
}

function openNoteEditComposer(note) {
  if (!note) return
  noteComposer.value = {
    open: true,
    mode: 'edit',
    quoteText: note.quoteText || '',
    content: note.content || '',
    noteId: note.id,
    selection: null,
  }
}

function closeNoteComposer() {
  noteComposer.value = { ...noteComposer.value, open: false }
}

function openNoteDeleteConfirm(note) {
  if (!note?.id) return
  noteDeleteTarget.value = {
    id: note.id,
    documentId: note.documentId || selectedDocument.value?.id || '',
    page: Number(note.page || 0),
    quoteText: String(note.quoteText || ''),
    content: String(note.content || ''),
  }
  noteDeleteStatus.value = 'idle'
  noteDeleteError.value = ''
  noteDeleteConfirmOpen.value = true
}

function closeNoteDeleteConfirm({ force = false } = {}) {
  if (!force && noteDeleteStatus.value === 'deleting') return
  noteDeleteConfirmOpen.value = false
  noteDeleteStatus.value = 'idle'
  noteDeleteError.value = ''
  noteDeleteTarget.value = {
    id: '',
    documentId: '',
    page: 0,
    quoteText: '',
    content: '',
  }
}

const noteDeleteTargetPreview = computed(() => {
  const target = noteDeleteTarget.value
  const source = String(target.quoteText || target.content || '').replace(/\s+/g, ' ').trim()
  if (source) return source.length > 180 ? `${source.slice(0, 180)}...` : source
  if (!target.page) return ''
  return locale.value === 'zh' ? `第${target.page}${ui.value.page}` : `${ui.value.page} ${target.page}`
})

async function submitNoteComposer(content) {
  if (noteComposerSaving.value) return
  const composer = noteComposer.value
  const doc = selectedDocument.value
  if (!doc || doc.id === 'empty') return
  noteComposerSaving.value = true
  try {
    if (composer.mode === 'edit') {
      const updated = await invoke('update_note', { input: { id: composer.noteId, content } })
      const idx = doc.notes.findIndex((note) => note.id === updated.id)
      if (idx >= 0) doc.notes.splice(idx, 1, updated)
    } else {
      const selected = composer.selection
      const created = await invoke('create_note', {
        input: {
          documentId: doc.id,
          page: selected?.page || activePage.value || 1,
          bboxList: selected?.bboxList || [],
          quoteText: selected?.text || '',
          content,
        },
      })
      doc.notes = [created, ...(doc.notes || [])]
      doc.notesLoaded = true
      lastSelection.value = null
      setRightPaneTab('notes')
      focusNote(created)
    }
    closeNoteComposer()
  } catch (err) {
    console.warn('Failed to save note', err)
  } finally {
    noteComposerSaving.value = false
  }
}

async function confirmDeleteNote() {
  if (noteDeleteStatus.value === 'deleting') return
  const target = noteDeleteTarget.value
  if (!target.id) return
  const doc = allDocs.value.find((item) => item.id === target.documentId) || selectedDocument.value
  if (!doc || doc.id === 'empty') return
  noteDeleteStatus.value = 'deleting'
  noteDeleteError.value = ''
  try {
    await invoke('delete_note', { input: { id: target.id } })
    doc.notes = (doc.notes || []).filter((item) => item.id !== target.id)
    if (activeNoteId.value === target.id) {
      activeNoteId.value = ''
      activeHighlight.value = null
    }
    closeNoteDeleteConfirm({ force: true })
  } catch (err) {
    console.warn('Failed to delete note', err)
    noteDeleteStatus.value = 'failed'
    noteDeleteError.value = String(err?.message || err || '')
  }
}

function focusNote(note) {
  if (!note) return
  activeNoteId.value = note.id
  activePage.value = note.page
  activeHighlight.value = {
    page: note.page,
    bboxList: note.bboxList || [],
  }
}

function isChatMessage(message) {
  return Boolean(message && typeof message === 'object')
}

function createWelcomeMessage(documentId) {
  return {
    id: `welcome-${documentId}-${Date.now()}`,
    role: 'assistant',
    content: {
      en: '',
      zh: '',
    },
    citations: [],
  }
}

function messageFingerprint(message) {
  if (!isChatMessage(message)) return ''
  const content = typeof message.content === 'object'
    ? (message.content.en || message.content.zh || '')
    : String(message.content || '')
  return `${message.role || ''}:${String(content).trim()}`
}

function chatTurnIdFromMessage(message) {
  const turnId = String(message?.turnId || '').trim()
  if (turnId) return turnId
  const id = String(message?.id || '')
  return id.endsWith(':user') || id.endsWith(':assistant')
    ? id.slice(0, id.lastIndexOf(':'))
    : ''
}

// Edit the last user question and re-ask: drop that turn from the UI and from
// the persisted store, then resend the edited text (preserving the turn's
// @-mentions). Persisted turn id comes from `turnId` (loaded) or `activityEventId`
// (sent this session).
async function handleEditResend({ messageId, text } = {}) {
  const newText = String(text || '').trim()
  if (!messageId || !newText) return
  const session = activeSession.value
  if (!session || !Array.isArray(session.messages)) return
  const messages = session.messages
  const userIndex = messages.findIndex((item) => item.id === messageId && item.role === 'user')
  if (userIndex === -1) return
  const mentions = Array.isArray(messages[userIndex].mentionedDocumentIds)
    ? [...messages[userIndex].mentionedDocumentIds]
    : []
  const turnIds = [...new Set(
    messages
      .slice(userIndex)
      .map((item) => chatTurnIdFromMessage(item) || String(item?.activityEventId || '').trim())
      .filter(Boolean),
  )]
  // Drop the turn (and anything after it) from the UI.
  session.messages = messages.slice(0, userIndex)
  // Delete the persisted turn(s); best-effort.
  if (turnIds.length) {
    invoke('clear_chat_turns', { input: { sessionId: session.id, turnIds } })
      .catch((err) => console.warn(ui.value.clearChatHistoryFailed, err))
  }
  // Re-ask with the edited question.
  await handleSend({ text: newText, mentionedDocIds: mentions, ignoreSelection: true })
}

function openClearChatHistoryConfirm() {
  const session = activeSession.value
  if (!session) return
  clearChatError.value = ''
  clearChatStatus.value = 'idle'
  clearChatTargetSessionId.value = session.id
  clearChatTargetTitle.value = sessionTabTitle(session)
  clearChatConfirmOpen.value = true
}

function closeClearChatHistoryConfirm() {
  clearChatConfirmOpen.value = false
  clearChatError.value = ''
  clearChatTargetSessionId.value = ''
  clearChatTargetTitle.value = ''
}

function resetClearChatConfirm() {
  clearChatConfirmOpen.value = false
  clearChatStatus.value = 'idle'
  clearChatError.value = ''
  clearChatTargetSessionId.value = ''
  clearChatTargetTitle.value = ''
}

function resetRemoveWorkspaceRootConfirm() {
  removeWorkspaceRootConfirmOpen.value = false
  removeWorkspaceRootStatus.value = 'idle'
  removeWorkspaceRootError.value = ''
  removeWorkspaceRootTarget.value = {
    id: '',
    name: '',
    path: '',
    docCount: 0,
  }
}

function openRemoveWorkspaceRootConfirm(workspaceRoot = null) {
  if (!workspaceRoot?.id || workspaceStatus.value === 'scanning') return
  removeWorkspaceRootError.value = ''
  removeWorkspaceRootStatus.value = 'idle'
  removeWorkspaceRootTarget.value = {
    id: workspaceRoot.id,
    name: String(workspaceRoot.name?.en || workspaceRoot.name?.zh || workspaceRoot.path || ''),
    path: workspaceRoot.path || '',
    docCount: (workspaceRoot.folders || []).reduce((count, folder) => {
      count += Array.isArray(folder.docs) ? folder.docs.length : 0
      return count
    }, 0),
  }
  removeWorkspaceRootConfirmOpen.value = true
}

function closeRemoveWorkspaceRootConfirm() {
  resetRemoveWorkspaceRootConfirm()
}

function confirmRemoveWorkspaceRoot() {
  const target = removeWorkspaceRootTarget.value
  if (!target?.id || removeWorkspaceRootStatus.value === 'removing') return
  removeWorkspaceRootStatus.value = 'removing'
  removeWorkspaceRootError.value = ''

  void (async () => {
    try {
      await invoke('remove_workspace_root', { rootId: target.id })

      const removedRootId = target.id
      let removedIndex = workspace.roots.findIndex((item) => item.id === removedRootId)
      if (removedIndex < 0 && String(removedRootId) !== String(target.path)) {
        removedIndex = workspace.roots.findIndex((item) => item.path === target.path)
      }
      if (removedIndex >= 0) {
        workspace.roots.splice(removedIndex, 1)
      }

      // Drop the removed root from the persisted collapse map so it doesn't
      // accumulate stale keys.
      const staleKeys = [removedRootId, target.path].map(String).filter(Boolean)
      if (staleKeys.some((key) => key in (collapsedRoots.value || {}))) {
        const next = { ...collapsedRoots.value }
        staleKeys.forEach((key) => { delete next[key] })
        collapsedRoots.value = next
      }

      if (!workspace.roots.length) {
        selectedDocId.value = ''
        inlineTranslateOpen.value = false
        resetRemoveWorkspaceRootConfirm()
        return
      }

      if (!allDocs.value.some((doc) => doc.id === selectedDocId.value)) {
        selectedDocId.value = ''
      }

      if (selectedDocId.value) {
        loadChatHistoryForDocument(selectedDocId.value)
      }

      resetRemoveWorkspaceRootConfirm()
    } catch (err) {
      removeWorkspaceRootStatus.value = 'failed'
      removeWorkspaceRootError.value = err?.message || String(err)
    }
  })()
}

function openDeleteDocumentConfirm(doc = null) {
  if (!doc?.id || doc.id === 'empty') return
  documentDeleteError.value = ''
  documentDeleteStatus.value = 'idle'
  documentDeleteTarget.value = {
    id: doc.id,
    title: String(doc.shortTitle || doc.title || ''),
    isNote: doc.source === 'note' || String(doc.id).startsWith('note-'),
  }
  documentDeleteConfirmOpen.value = true
}

function closeDeleteDocumentConfirm() {
  if (documentDeleteStatus.value === 'deleting') return
  documentDeleteConfirmOpen.value = false
  documentDeleteStatus.value = 'idle'
  documentDeleteError.value = ''
  documentDeleteTarget.value = { id: '', title: '', isNote: false }
}

// Remove a document from the in-memory workspace tree after the backend deletes
// it. Returns true when found and spliced.
function removeDocFromWorkspaceState(documentId) {
  for (const root of workspace.roots) {
    for (const folder of root.folders || []) {
      const index = (folder.docs || []).findIndex((item) => item.id === documentId)
      if (index >= 0) {
        folder.docs.splice(index, 1)
        return true
      }
    }
  }
  return false
}

function confirmDeleteDocument() {
  const target = documentDeleteTarget.value
  if (!target?.id || documentDeleteStatus.value === 'deleting') return
  documentDeleteStatus.value = 'deleting'
  documentDeleteError.value = ''

  void (async () => {
    try {
      await invoke('delete_document', { documentId: target.id })
      removeDocFromWorkspaceState(target.id)
      openTabs.value = openTabs.value.filter((id) => id !== target.id)
      // Drop per-document caches so a future doc reusing nothing lingers.
      precipitationQueued.delete(target.id)
      visualIndexRuns.delete(target.id)
      // If the deleted document was active, fall back to the first remaining one.
      if (selectedDocId.value === target.id) {
        selectedDocId.value = ''
        if (selectedDocId.value) {
          loadChatHistoryForDocument(selectedDocId.value)
        }
      }
      documentDeleteConfirmOpen.value = false
      documentDeleteStatus.value = 'idle'
      documentDeleteTarget.value = { id: '', title: '', isNote: false }
    } catch (err) {
      documentDeleteStatus.value = 'failed'
      documentDeleteError.value = err?.message || String(err)
    }
  })()
}

function clearChatTurnsWithTimeout(sessionId, turnIds) {
  const clearPromise = invoke('clear_chat_turns', {
    input: {
      sessionId,
      turnIds,
    },
  })
  let timeoutId = 0
  const timeoutPromise = new Promise((_, reject) => {
    timeoutId = window.setTimeout(() => {
      const error = new Error('Local database is busy; persistence cleanup will continue in the background.')
      error.name = 'ClearChatHistoryTimeout'
      reject(error)
    }, CLEAR_CHAT_HISTORY_TIMEOUT_MS)
  })
  clearPromise.catch((err) => {
    console.warn(ui.value.clearChatHistoryFailed, err)
  })
  return Promise.race([clearPromise, timeoutPromise])
    .finally(() => {
      if (timeoutId) window.clearTimeout(timeoutId)
    })
}

async function finishClearChatPersistence(session, previousMessages, clearedEventIds, turnIds, clearWelcomeId) {
  try {
    await clearChatTurnsWithTimeout(session.id, turnIds)
  } catch (err) {
    if (err?.name === 'ClearChatHistoryTimeout') {
      console.warn(ui.value.clearChatHistoryFailed, err)
      return
    }
    const stillShowingClearPlaceholder = session.messages?.length === 1 && session.messages[0]?.id === clearWelcomeId
    if (stillShowingClearPlaceholder) session.messages = previousMessages
    clearedEventIds.forEach((eventId) => clearedActivityEventIds.delete(eventId))
    clearChatTargetSessionId.value = session.id
    clearChatTargetTitle.value = sessionTabTitle(session)
    clearChatStatus.value = 'failed'
    clearChatError.value = err?.message || String(err)
    clearChatConfirmOpen.value = true
  }
}

function confirmClearChatHistory() {
  const session = chatSessions.get(clearChatTargetSessionId.value)
  if (!session) {
    closeClearChatHistoryConfirm()
    return
  }
  if (clearChatStatus.value === 'clearing') return
  clearChatStatus.value = 'clearing'
  clearChatError.value = ''
  session.chatHistoryClearGeneration = Number(session.chatHistoryClearGeneration || 0) + 1
  const previousMessages = Array.isArray(session.messages) ? session.messages.filter(isChatMessage) : []
  const clearedEventIds = previousMessages
    .map((message) => message.activityEventId)
    .filter(Boolean)
  const turnIds = [...new Set(previousMessages.map(chatTurnIdFromMessage).filter(Boolean))]
  clearedEventIds.forEach((eventId) => {
    clearedActivityEventIds.set(eventId, session.id)
    clearAssistantStreamState(eventId)
  })
  session.messages = [createWelcomeMessage(session.id)]
  const clearWelcomeId = session.messages[0]?.id || ''
  session.chatHistoryLoaded = true
  session.chatHistoryLoading = null
  if (activeSessionId.value === session.id) {
    activeCitationId.value = ''
    activeBlockId.value = ''
    activeHighlight.value = null
  }
  resetClearChatConfirm()
  void finishClearChatPersistence(session, previousMessages, clearedEventIds, turnIds, clearWelcomeId)
}

function applyAskDocumentError(eventId, errorMessage) {
  if (clearedActivityEventIds.has(eventId)) {
    clearAssistantStreamState(eventId)
    clearedActivityEventIds.delete(eventId)
    return
  }
  const message = findMessageByActivityEventId(eventId)
  if (!message) return
  clearAssistantStreamState(eventId)
  const error = errorMessage || 'Unknown error'
  message.content = {
    en: `${messages.en.chatFailed}: ${error}`,
    zh: `${messages.zh.chatFailed}: ${error}`,
  }
  message.status = 'failed'
  if (!Array.isArray(message.activityEvents)) message.activityEvents = []
  message.activityEvents.push({
    type: 'error',
    step: 'error',
    status: 'error',
    title: locale.value === 'zh' ? 'Agent 失败' : 'Agent failed',
    summary: error,
    detail: error,
    ts: Date.now(),
  })
}

function handleAgentActivity(payload) {
  const eventId = payload?.eventId
  const event = payload?.event
  if (!eventId || !event) return
  if (clearedActivityEventIds.has(eventId)) return
  const message = findMessageByActivityEventId(eventId)
  if (!message) return
  if (!Array.isArray(message.activityEvents)) message.activityEvents = []
  message.activityEvents.push(event)
}

function handleAnswerDelta(payload) {
  const eventId = payload?.eventId
  const delta = String(payload?.delta || '')
  if (!eventId || !delta) return
  if (clearedActivityEventIds.has(eventId)) return
  chatStreamDebug('delta event received', {
    eventId,
    deltaLength: delta.length,
    hasMessage: Boolean(findMessageByActivityEventId(eventId)),
    preview: delta.slice(0, 24),
  })
  queueAssistantStreamText(eventId, delta)
}

function handleReasoningDelta(payload) {
  const eventId = payload?.eventId
  const delta = String(payload?.delta || '')
  if (!eventId || !delta) return
  if (clearedActivityEventIds.has(eventId)) return
  const message = findMessageByActivityEventId(eventId)
  if (!message) return
  message.reasoningContent = `${message.reasoningContent || ''}${delta}`
}

function canOpenTranslatedView(doc = selectedDocument.value) {
  const translation = doc?.translation
  if (!translation) return false
  if (['pending', 'queued', 'running', 'partial', 'succeeded', 'failed'].includes(translation.status)) {
    return true
  }
  return translation.status === 'canceled' && Boolean(translation.monoPdfPath || translation.dualPdfPath)
}

function resetDocumentTranslationState(doc, lang = translationLang.value, options = {}) {
  if (!doc?.translation) return
  if (options.cancelLoads !== false) {
    cancelQueuedTranslationPageLoads()
  }
  doc.translation.status = 'idle'
  doc.translation.progress = 0
  doc.translation.total = 0
  doc.translation.failedBlocks = 0
  doc.translation.lang = lang
  doc.translation.error = ''
  doc.translation.jobId = ''
  doc.translation.providerKey = ''
  doc.translation.phase = ''
  doc.translation.currentPage = 0
  doc.translation.pdfJobId = ''
  doc.translation.pdfStatus = 'idle'
  doc.translation.pdfProgressPercent = 0
  doc.translation.monoPdfPath = ''
  doc.translation.dualPdfPath = ''
  doc.translation.pdfArtifactScope = ''
  doc.translation.pdfArtifactPages = ''
  doc.translation.partialArtifacts = {}
  doc.translation.cached = false
  doc.translation.pages = {}
}

function resetTranslationAfterIndexChange(doc) {
  if (!doc?.translation) return
  const lang = doc.translation.lang || translationLang.value
  const isSelected = doc.id === selectedDocId.value
  resetDocumentTranslationState(doc, lang, { cancelLoads: isSelected })
  if (isSelected && ['translated', 'dual'].includes(viewMode.value)) {
    viewMode.value = 'original'
  }
}

function setViewMode(mode) {
  if (['translated', 'dual'].includes(mode) && !canOpenTranslatedView()) return
  viewMode.value = mode
  if (mode !== 'dual') hoveredLinkedBlock.value = null
  if (['translated', 'dual'].includes(mode)) {
    loadActivePageTranslation()
  }
}

function normalizePositivePage(value, fallback = 1) {
  const page = Number(value)
  return Number.isFinite(page) && page > 0 ? Math.floor(page) : fallback
}

function normalizeVisiblePages(value, fallbackPage) {
  const pages = Array.isArray(value) ? value : []
  const seen = new Set()
  const normalized = []
  for (const item of pages) {
    const page = normalizePositivePage(item, 0)
    if (!page || seen.has(page)) continue
    seen.add(page)
    normalized.push(page)
  }
  if (!normalized.length && fallbackPage) normalized.push(fallbackPage)
  return normalized
}

function parsePdfArtifactPages(value, fallbackPage = 0) {
  const pages = String(value || '')
    .split(',')
    .flatMap((part) => {
      const token = part.trim()
      if (!token) return []
      const [startValue, endValue] = token.split('-').map((item) => Number(item.trim()))
      if (!Number.isFinite(startValue) || startValue <= 0) return []
      if (!Number.isFinite(endValue) || endValue <= startValue) return [Math.floor(startValue)]
      const range = []
      for (let page = Math.floor(startValue); page <= Math.floor(endValue); page += 1) {
        range.push(page)
      }
      return range
    })
  if (!pages.length && fallbackPage) pages.push(fallbackPage)
  return pages
}

function formatPdfTranslationError(message) {
  const text = String(message || '')
  if (locale.value !== 'zh' || !text.includes('Translation is unavailable')) return text
  if (text.includes('TLS connection failed')) {
    return '翻译不可用：Google Web / Microsoft(Bing) Web 网络检测未通过。检测到 TLS 连接失败，通常是当前网络、代理或服务端连接被中断导致。请检查代理/VPN，或切换到 LLM Provider。'
  }
  if (text.includes('Request timed out')) {
    return '翻译不可用：Google Web / Microsoft(Bing) Web 网络检测超时。请检查网络或代理，稍后重试，或切换到 LLM Provider。'
  }
  if (text.includes('Authentication') || text.includes('not configured')) {
    return '翻译不可用：翻译 Provider 鉴权失败或配置不完整。请检查 Provider 设置后重试。'
  }
  return '翻译不可用：Google Web / Microsoft(Bing) Web 均未通过网络检测。请检查网络/代理，或切换到 LLM Provider。'
}

// A scanned / image-only PDF has no text layer, so Babeldoc returns "contains no
// paragraphs" (code=BabeldocError). That isn't a real, retryable failure —
// translation simply doesn't apply. Detect it so we can show a calm "not
// available" notice instead of a scary error, and skip re-running the flow.
function isScannedTranslationError(message) {
  return /contains no paragraphs|no paragraphs|BabeldocError/i.test(String(message || ''))
}

function applyTranslationUnsupported(doc) {
  if (!doc?.translation) return
  const translation = doc.translation
  translation.status = 'unsupported'
  translation.pdfStatus = 'unsupported'
  translation.phase = 'unsupported'
  translation.error = ui.value.scannedPdfTranslationUnsupported
  translation.progress = 0
  translation.total = 0
  translation.failedBlocks = 0
  // Remember so a later click on Translate doesn't re-run the doomed flow.
  doc.translationUnsupported = true
}

// Map a backend index error to a user-facing message. Scanned/image-only PDFs
// surface the SCANNED_PDF_NO_TEXT sentinel (see pdf_index/mod.rs) which we turn
// into a localized, friendly explanation instead of a raw Pdfium error.
function formatIndexError(message) {
  const text = String(message || '')
  if (text.includes('SCANNED_PDF_NO_TEXT')) return ui.value.scannedPdfUnsupported
  return text
}

// TEMP perf probe: measures the synchronous click cost plus the duration of
// the next several animation frames. A long frame gap (>>16ms) right after the
// click reveals where the jank is (Vue DOM patch vs. browser layout/paint).
// Remove once the translate-open jank is diagnosed.
function probeTranslateFrames(label, t0) {
  window.__lfPerf = {}
  const syncMs = performance.now() - t0
  // Split the cost: Vue DOM patch (domPatch) vs. browser layout/paint
  // (patchToPaint). A big patchToPaint means the jank is rendering (e.g. the
  // PDF canvas reflowing to half width), not JavaScript.
  const tCall = performance.now()
  nextTick(() => {
    const domPatch = performance.now() - tCall
    requestAnimationFrame(() => {
      const patchToPaint = performance.now() - tCall - domPatch
      // eslint-disable-next-line no-console
      console.log(`[translate-perf] ${label} domPatch=${Math.round(domPatch)}ms patchToPaint=${Math.round(patchToPaint)}ms`)
    })
  })
  let last = performance.now()
  const frames = []
  let count = 0
  const tick = () => {
    const now = performance.now()
    frames.push(Math.round(now - last))
    last = now
    count += 1
    if (count < 16) {
      requestAnimationFrame(tick)
    } else {
      const perf = window.__lfPerf || {}
      const hot = Object.entries(perf)
        .map(([k, v]) => [k, v, Math.round(v.ms)])
        .sort((a, b) => b[2] - a[2])
        .map(([k, v, ms]) => `${k}:${ms}ms/${v.n}`)
        .join('  ')
      // eslint-disable-next-line no-console
      console.log(`[translate-perf] ${label} sync=${syncMs.toFixed(1)}ms frames(ms)=[${frames.join(', ')}] hot=[${hot}]`)
      window.__lfPerf = {}
    }
  }
  requestAnimationFrame(tick)
}

// TEMP perf helper: accumulate per-label synchronous time into window.__lfPerf
// so probeTranslateFrames can dump the hottest functions after a translate click.
function lfPerf(label, fn) {
  const start = performance.now()
  try {
    return fn()
  } finally {
    const store = (window.__lfPerf ||= {})
    const entry = (store[label] ||= { ms: 0, n: 0 })
    entry.ms += performance.now() - start
    entry.n += 1
  }
}

async function handleTranslationAction(viewerContext = {}) {
  const perfT0 = performance.now()
  const doc = selectedDocument.value
  if (!doc?.chatReady) return
  const translation = doc.translation
  if (translation.status === 'succeeded') {
    viewMode.value = viewMode.value === 'dual' ? 'original' : 'dual'
    if (viewMode.value === 'dual') rightCollapsed.value = true
    probeTranslateFrames('toggle-dual', perfT0)
    loadActivePageTranslation()
    return
  }

  if (['pending', 'queued', 'running', 'partial'].includes(translation.status)) {
    return
  }

  // Scanned / no-text-layer PDF: translation can't work. Show the calm notice in
  // the translation pane and don't run the flow at all (we learned this from a
  // prior attempt on this document).
  if (doc.translationUnsupported) {
    applyTranslationUnsupported(doc)
    viewMode.value = 'dual'
    rightCollapsed.value = true
    return
  }

  translation.error = ''
  translation.status = 'pending'
  translation.progress = 0
  translation.total = 0
  translation.failedBlocks = 0
  const canTryPdfSidecar = doc.source === 'local'
    && (!pdfTranslationRuntime.value.checked || pdfTranslationRuntime.value.ok)
  translation.phase = canTryPdfSidecar ? 'checking_provider' : 'queued'
  const priorityPage = normalizePositivePage(
    viewerContext.currentPage || activePage.value || doc.currentPage,
    1,
  )
  const visiblePages = normalizeVisiblePages(viewerContext.visiblePages, priorityPage)
  const sidecarPageCount = normalizePositivePage(
    viewerContext.pageCount || doc.pageCount,
    0,
  ) || undefined
  translation.currentPage = priorityPage
  translation.pdfJobId = ''
  translation.pdfStatus = 'queued'
  translation.pdfProgressPercent = 0
  translation.monoPdfPath = ''
  translation.dualPdfPath = ''
  translation.pdfArtifactScope = ''
  translation.pdfArtifactPages = ''
  translation.cached = false
  translation.pages = {}
  viewMode.value = 'dual'
  rightCollapsed.value = true
  probeTranslateFrames('first-translate', perfT0)

  try {
    const usePdfSidecar = canTryPdfSidecar
    const __i0 = performance.now()
    const result = usePdfSidecar
      ? await invoke('start_pdf_translation', {
        input: {
          documentId: doc.id,
          targetLang: translationLang.value,
          sourceLang: 'en',
          provider: translationProvider.value,
          artifactMode: 'mono',
          priorityPage,
          visiblePages,
          nearbyPageRadius: 1,
          pageCount: sidecarPageCount,
          forceRefresh: false,
        },
      })
      : await invoke('start_document_translation', {
        input: {
          documentId: doc.id,
          targetLang: translationLang.value,
          provider: translationProvider.value,
          scope: 'document',
          priorityPage,
          forceRefresh: false,
        },
      })
    // TEMP perf: invoke duration + serialized result size
    try {
      // eslint-disable-next-line no-console
      console.log(`[translate-perf] ${usePdfSidecar ? 'start_pdf' : 'start_doc'} invoke=${Math.round(performance.now() - __i0)}ms resultBytes=${JSON.stringify(result).length}`)
    } catch (e) { /* ignore */ }
    if (selectedDocId.value !== doc.id) return
    const __a0 = performance.now()
    if (usePdfSidecar) {
      if (!isPdfTranslationJobRelevant(doc, result)) return
      applyPdfTranslationJobResult(doc, result)
    } else {
      if (!isTranslationJobRelevant(doc, result)) return
      applyTranslationJobResult(doc, result)
    }
    // eslint-disable-next-line no-console
    console.log(`[translate-perf] applyResult=${Math.round(performance.now() - __a0)}ms`)
    if (canOpenTranslatedView(doc) && ['translated', 'dual'].includes(viewMode.value)) {
      scheduleActivePageTranslationLoad(0)
    }
  } catch (err) {
    const raw = err?.message || String(err)
    if (isScannedTranslationError(raw)) {
      applyTranslationUnsupported(doc)
    } else {
      translation.status = 'failed'
      translation.error = formatPdfTranslationError(raw)
    }
  }
}

async function cancelTranslation() {
  const doc = selectedDocument.value
  if (!doc) return
  const translation = doc.translation
  clearInterval(translationTimer)
  const jobId = translation.jobId
  const pdfJobId = translation.pdfJobId
  const translationJobActive = ['pending', 'queued', 'running', 'partial'].includes(translation.status)
  if (!translationJobActive || (!jobId && !pdfJobId)) return
  if (translationJobActive) {
    translation.status = 'canceled'
    translation.pdfStatus = 'canceled'
  }
  try {
    await Promise.resolve(
      pdfJobId && translationJobActive
        ? invoke('cancel_pdf_translation', { input: { jobId: pdfJobId } })
        : jobId && translationJobActive
        ? invoke('cancel_translation_job', { input: { jobId } })
        : Promise.resolve(),
    )
  } catch (err) {
    translation.error = err?.message || String(err)
  }
}

function applyTranslationJobResult(doc, result) {
  if (!doc || !result) return
  doc.translation.jobId = result.jobId || doc.translation.jobId || ''
  doc.translation.status = result.status || doc.translation.status || 'idle'
  doc.translation.progress = Number(result.translatedBlocks || 0)
  doc.translation.total = Number(result.totalBlocks || 0)
  doc.translation.failedBlocks = Number(result.failedBlocks || 0)
  doc.translation.providerKey = result.providerKey || doc.translation.providerKey || ''
  doc.translation.phase = result.phase || doc.translation.phase || ''
  doc.translation.currentPage = Number(result.currentPage || doc.translation.currentPage || 0)
  doc.translation.error = result.error || ''
  if (doc.translation.status === 'failed' && isScannedTranslationError(doc.translation.error)) {
    applyTranslationUnsupported(doc)
  }
}

function applyPdfTranslationJobResult(doc, result) {
  if (!doc || !result) return
  doc.translation.pdfJobId = result.jobId || doc.translation.pdfJobId || ''
  doc.translation.jobId = result.jobId || doc.translation.jobId || ''
  doc.translation.status = result.status || doc.translation.status || 'idle'
  doc.translation.pdfStatus = result.status || doc.translation.pdfStatus || 'idle'
  doc.translation.progress = Number(result.progressPercent || 0)
  doc.translation.total = 100
  doc.translation.failedBlocks = 0
  doc.translation.providerKey = result.providerKey || doc.translation.providerKey || ''
  doc.translation.phase = result.phase || doc.translation.phase || ''
  doc.translation.currentPage = Number(result.currentPage || doc.translation.currentPage || 0)
  doc.translation.pdfProgressPercent = Number(result.progressPercent || 0)
  doc.translation.monoPdfPath = result.monoPdfPath || doc.translation.monoPdfPath || ''
  doc.translation.dualPdfPath = result.dualPdfPath || doc.translation.dualPdfPath || ''
  doc.translation.pdfArtifactScope = result.artifactScope || doc.translation.pdfArtifactScope || ''
  doc.translation.pdfArtifactPages = result.artifactPages || doc.translation.pdfArtifactPages || ''
  if (result.artifactScope === 'partial' && result.monoPdfPath) {
    const artifactPages = parsePdfArtifactPages(
      result.artifactPages,
      Number(result.currentPage || doc.translation.currentPage || 0),
    )
    const nextArtifacts = { ...(doc.translation.partialArtifacts || {}) }
    for (const page of artifactPages) {
      nextArtifacts[page] = {
        monoPdfPath: result.monoPdfPath || '',
        dualPdfPath: result.dualPdfPath || '',
        artifactPages: result.artifactPages || String(page),
      }
    }
    doc.translation.partialArtifacts = nextArtifacts
  }
  doc.translation.cached = Boolean(result.cached)
  doc.translation.error = result.error || ''
  if (doc.translation.status === 'failed' && isScannedTranslationError(doc.translation.error)) {
    applyTranslationUnsupported(doc)
  }
}

function isTranslationJobRelevant(doc, payload) {
  if (!doc?.translation || !payload) return false
  if (payload.documentId && payload.documentId !== doc.id) return false
  if (payload.targetLang && payload.targetLang !== doc.translation.lang) return false
  if (doc.translation.jobId && payload.jobId && payload.jobId !== doc.translation.jobId) {
    return false
  }
  if (!doc.translation.jobId && !['pending', 'queued', 'running'].includes(doc.translation.status)) {
    return false
  }
  return true
}

function isPdfTranslationJobRelevant(doc, payload) {
  if (!doc?.translation || !payload) return false
  if (payload.documentId && payload.documentId !== doc.id) return false
  if (payload.targetLang && payload.targetLang !== doc.translation.lang) return false
  if (doc.translation.pdfJobId && payload.jobId && payload.jobId !== doc.translation.pdfJobId) {
    return false
  }
  if (!doc.translation.pdfJobId && !['pending', 'queued', 'running', 'partial'].includes(doc.translation.status)) {
    return false
  }
  return true
}

function handleTranslationJobEvent(payload) {
  const documentId = payload?.documentId
  if (!documentId) return
  const doc = allDocs.value.find((item) => item.id === documentId)
  if (!doc) return
  if (!isTranslationJobRelevant(doc, payload)) return
  applyTranslationJobResult(doc, payload)
  if (doc.id === selectedDocId.value) {
    const eventPage = Number(payload?.currentPage || 0)
    const status = payload?.status || ''
    if (['translated', 'dual'].includes(viewMode.value)) {
      if (eventPage) {
        scheduleTranslationPageLoad(documentId, payload?.jobId || '', eventPage)
      } else if (['partial', 'succeeded', 'failed', 'canceled'].includes(status)) {
        scheduleActivePageTranslationLoad()
      }
      return
    }
    const shouldRefreshPage = !eventPage
      || eventPage === Number(activePage.value)
      || ['partial', 'succeeded', 'failed', 'canceled'].includes(status)
    if (shouldRefreshPage) scheduleActivePageTranslationLoad()
  }
}

function handlePdfTranslationEvent(payload) {
  const documentId = payload?.documentId
  if (!documentId) return
  const doc = allDocs.value.find((item) => item.id === documentId)
  if (!doc) return
  if (!isPdfTranslationJobRelevant(doc, payload)) return
  applyPdfTranslationJobResult(doc, payload)
  if (doc.id === selectedDocId.value && ['translated', 'dual'].includes(viewMode.value)) {
    scheduleActivePageTranslationLoad()
  }
}

function cancelQueuedTranslationPageLoads() {
  for (const timer of translationSinglePageRefreshTimers.values()) {
    clearTimeout(timer)
  }
  translationSinglePageRefreshTimers.clear()
  translationPageLoadsInFlight.clear()
}

function scheduleTranslationPageLoad(documentId, jobId, pageNo, delay = 350) {
  const normalizedPage = Number(pageNo)
  if (!documentId || !normalizedPage) return
  const key = `${documentId}:${jobId || 'unknown'}:${normalizedPage}`
  const existing = translationSinglePageRefreshTimers.get(key)
  if (existing) clearTimeout(existing)
  const timer = window.setTimeout(() => {
    translationSinglePageRefreshTimers.delete(key)
    loadPageTranslation(normalizedPage, {
      requireActivePage: false,
      expectedDocumentId: documentId,
      expectedJobId: jobId || '',
    })
  }, delay)
  translationSinglePageRefreshTimers.set(key, timer)
}

function scheduleActivePageTranslationLoad(delay = 350) {
  clearTimeout(translationPageRefreshTimer)
  translationPageRefreshTimer = setTimeout(() => {
    translationPageRefreshTimer = null
    loadActivePageTranslation()
  }, delay)
}

async function loadActivePageTranslation() {
  const doc = selectedDocument.value
  if (!doc || doc.id === 'empty' || !canOpenTranslatedView(doc)) return
  const pageNo = Number(activePage.value || doc.currentPage || 1)
  if (!pageNo) return
  await loadPageTranslation(pageNo, { requireActivePage: true })
}

async function handleTranslationPageRequest(pageNo) {
  const doc = selectedDocument.value
  const normalizedPage = Number(pageNo)
  if (!doc || doc.id === 'empty' || !normalizedPage || !canOpenTranslatedView(doc)) return
  const cached = doc.translation.pages?.[normalizedPage]
  if (isTranslationPageTerminal(cached)) return
  await loadPageTranslation(normalizedPage, {
    requireActivePage: false,
    expectedDocumentId: doc.id,
    expectedJobId: doc.translation.jobId,
  })
}

async function handlePdfTranslationPagesRequest(payload = {}) {
  const doc = selectedDocument.value
  if (!doc || doc.id === 'empty') return
  const jobId = payload.jobId || doc.translation.pdfJobId
  if (!jobId || jobId !== doc.translation.pdfJobId) return
  const pages = normalizeVisiblePages(payload.pages, 0)
  if (!pages.length) return
  try {
    await invoke('request_pdf_translation_pages', {
      input: {
        jobId,
        pages,
      },
    })
  } catch (err) {
    doc.translation.error = err?.message || String(err)
  }
}

function isTranslationPageTerminal(pageTranslation) {
  return ['succeeded', 'partial', 'failed'].includes(pageTranslation?.status)
}

async function loadPageTranslation(pageNo, options = {}) {
  const requireActivePage = options.requireActivePage !== false
  const doc = selectedDocument.value
  if (!doc || doc.id === 'empty' || !canOpenTranslatedView(doc)) return
  if (options.expectedDocumentId && doc.id !== options.expectedDocumentId) return
  if (options.expectedJobId && doc.translation.jobId !== options.expectedJobId) return
  if (!pageNo) return
  const loadKey = [
    doc.id,
    doc.translation.jobId || 'no-job',
    translationLang.value,
    translationProvider.value,
    pageNo,
  ].join(':')
  if (translationPageLoadsInFlight.has(loadKey)) return
  translationPageLoadsInFlight.add(loadKey)
  try {
    const __g0 = performance.now()
    const result = await invoke('get_page_translation', {
      input: {
        documentId: doc.id,
        pageNo,
        targetLang: translationLang.value,
        provider: translationProvider.value,
        providerKey: doc.translation.providerKey || '',
      },
    })
    // TEMP perf
    try {
      // eslint-disable-next-line no-console
      console.log(`[translate-perf] get_page_translation(p${pageNo}) invoke=${Math.round(performance.now() - __g0)}ms resultBytes=${JSON.stringify(result).length}`)
    } catch (e) { /* ignore */ }
    if (selectedDocId.value !== doc.id) return
    if (options.expectedDocumentId && selectedDocId.value !== options.expectedDocumentId) return
    if (options.expectedJobId && doc.translation.jobId !== options.expectedJobId) return
    if (requireActivePage && Number(activePage.value) !== pageNo) return
    const __w0 = performance.now()
    doc.translation.pages = {
      ...(doc.translation.pages || {}),
      [pageNo]: {
        ...result,
      },
    }
    // eslint-disable-next-line no-console
    console.log(`[translate-perf] writePage(p${pageNo})=${Math.round(performance.now() - __w0)}ms`)
  } catch (err) {
    doc.translation.error = err?.message || String(err)
  } finally {
    translationPageLoadsInFlight.delete(loadKey)
  }
}

function toggleInlineTranslate() {
  inlineTranslateOpen.value = !inlineTranslateOpen.value
}

function setPage(page, blockId = '', source = null) {
  activePage.value = page
  if (source) hoveredLinkedBlock.value = null
  if (blockId) {
    activeBlockId.value = blockId
    activeCitationId.value = ''
  }
  if (source?.bboxList?.length) {
    activeHighlight.value = createHighlight({
      page,
      bboxList: source.bboxList,
    })
  } else if (source?.clearHighlight) {
    activeHighlight.value = null
  }
  if (selectedDocument.value?.source === 'local') {
    selectedDocument.value.currentPage = page
    invoke('update_document_reading_state', {
      input: {
        documentId: selectedDocument.value.id,
        page,
        zoom: 1.18,
      },
    }).catch((err) => {
      console.warn('Failed to save reading state', err)
    })
  }
}

function setLinkedBlockHover(block = null) {
  hoveredLinkedBlock.value = normalizeLinkedBlockHover(block)
}

function handleDocumentLoaded(payload) {
  if (!selectedDocument.value || selectedDocument.value.id === 'empty') return
  selectedDocument.value.pageCount = payload.pageCount
  selectedDocument.value.chatReady = selectedDocument.value.indexStatus === 'indexed'
    && selectedDocument.value.indexVersion === selectedDocument.value.currentIndexVersion
  selectedDocument.value.translation.error = ''
  if (shouldQueueBackendIndex(selectedDocument.value)) {
    enqueueBackendDocumentIndex(selectedDocument.value)
  }
}

function handleDocumentLoadFailed(payload) {
  if (!selectedDocument.value || selectedDocument.value.id === 'empty') return
  selectedDocument.value.chatReady = false
  selectedDocument.value.status = 'stale'
  selectedDocument.value.statusTone = 'danger'
  selectedDocument.value.indexStatus = 'stale'
  selectedDocument.value.indexVersion = 0
  selectedDocument.value.treeReady = false
  selectedDocument.value.indexProgress = indexProgressState(0, '', '')
  selectedDocument.value.translation.error = payload?.error || ''
}

function handleDocumentIndexStarted() {
  if (!selectedDocument.value || selectedDocument.value.id === 'empty') return
  resetTranslationAfterIndexChange(selectedDocument.value)
  selectedDocument.value.status = 'indexing'
  selectedDocument.value.statusTone = 'warning'
  selectedDocument.value.indexStatus = 'indexing'
  selectedDocument.value.indexVersion = 0
  selectedDocument.value.treeReady = false
  selectedDocument.value.chatReady = false
  selectedDocument.value.indexProgress = indexProgressState(1, 'starting', '')
}

function handleDocumentIndexComplete(payload) {
  if (!selectedDocument.value || selectedDocument.value.id === 'empty') return
  resetTranslationAfterIndexChange(selectedDocument.value)
  delete selectedDocument.value.indexBeforeReindex
  selectedDocument.value.pageCount = payload.pageCount
  selectedDocument.value.indexVersion = payload.indexVersion || selectedDocument.value.currentIndexVersion
  selectedDocument.value.status = 'indexed'
  selectedDocument.value.statusTone = 'success'
  selectedDocument.value.indexStatus = 'indexed'
  selectedDocument.value.treeReady = Boolean(payload.treeReady ?? true)
  selectedDocument.value.chatReady = selectedDocument.value.treeReady
  selectedDocument.value.indexProgress = indexProgressState(100, 'complete', '')
  selectedDocument.value.visualIndexStatus = payload.visualIndexStatus || 'pending'
  selectedDocument.value.visualIndexVersion = Number(payload.visualIndexVersion || 0)
  selectedDocument.value.visualIndexError = payload.visualIndexError || ''
  workspaceError.value = ''
  scheduleDocumentVisualIndex(selectedDocument.value)
}

function handleDocumentIndexFailed(payload) {
  if (!selectedDocument.value || selectedDocument.value.id === 'empty') return
  resetTranslationAfterIndexChange(selectedDocument.value)
  selectedDocument.value.status = 'stale'
  selectedDocument.value.statusTone = 'danger'
  selectedDocument.value.indexStatus = 'stale'
  selectedDocument.value.indexVersion = 0
  selectedDocument.value.treeReady = false
  selectedDocument.value.indexProgress = indexProgressState(0, 'failed', '')
  selectedDocument.value.visualIndexStatus = 'pending'
  selectedDocument.value.visualIndexVersion = 0
  selectedDocument.value.visualIndexError = ''
  selectedDocument.value.translation.error = payload?.error || ''
  workspaceError.value = payload?.error ? formatIndexError(payload.error) : ''
}

function scheduleDocumentVisualIndex(doc) {
  if (!doc || doc.id === 'empty') return
  // Only PDFs have a visual (TSR) layer — notes/clips/md and Office sources don't.
  if (doc.contentType && doc.contentType !== 'pdf') return
  if (doc.source !== 'local' || doc.indexStatus !== 'indexed' || !doc.treeReady) return
  if (Number(doc.indexVersion || 0) !== Number(doc.currentIndexVersion || 0)) return
  if (doc.visualIndexStatus === 'succeeded'
    && Number(doc.visualIndexVersion || 0) === Number(doc.currentIndexVersion || 0)) {
    return
  }
  if (visualIndexRuns.has(doc.id)) return
  visualIndexRuns.add(doc.id)
  doc.visualIndexStatus = 'running'
  doc.visualIndexError = ''
  invoke('enqueue_document_visual_index', { documentId: doc.id })
    .then((result) => {
      const target = allDocs.value.find((item) => item.id === result.documentId) || doc
      target.visualIndexStatus = result.status || 'queued'
      target.visualIndexVersion = Number(result.version || target.currentIndexVersion || 0)
      target.visualIndexError = result.error || ''
    })
    .catch((err) => {
      const target = allDocs.value.find((item) => item.id === doc.id) || doc
      target.visualIndexStatus = 'failed'
      target.visualIndexError = err?.message || String(err)
      // Visual indexing (tables/figures) is an enhancement on top of a usable text
      // index — a failure must NOT raise the global red error bar. It's recorded
      // on the document for a quiet indicator only.
      visualIndexRuns.delete(doc.id)
    })
}

// ---- Knowledge precipitation (Stream 1) front-end orchestration ----

async function loadKnowledgeCard(documentId) {
  if (!documentId || documentId === 'empty') {
    knowledgeCard.value = null
    relatedDocs.value = []
    relatedLinks.value = []
    return
  }
  try {
    knowledgeCard.value = await invoke('get_document_knowledge', { documentId })
  } catch {
    knowledgeCard.value = null
  }
  loadRelatedDocuments(documentId)
}

async function loadRelatedDocuments(documentId) {
  if (!documentId || documentId === 'empty') {
    relatedDocs.value = []
    relatedLinks.value = []
    return
  }
  try {
    const payload = await invoke('get_related_documents', { documentId })
    // Guard against a stale response after the user switched documents.
    if (selectedDocId.value === documentId) {
      relatedDocs.value = Array.isArray(payload?.related) ? payload.related : []
      relatedLinks.value = Array.isArray(payload?.links) ? payload.links : []
    }
  } catch {
    // Same stale guard: a late rejection for a previous document must not wipe
    // the current document's related panel.
    if (selectedDocId.value !== documentId) return
    relatedDocs.value = []
    relatedLinks.value = []
  }
}

async function reprecipitateCurrentDocument() {
  const documentId = selectedDocId.value
  if (!documentId || documentId === 'empty') return
  const { providerId, modelKey } = parseChatModelOptionId(selectedChatModelId.value)
  knowledgeByDoc[documentId] = { ...(knowledgeByDoc[documentId] || {}), status: 'running' }
  if (knowledgeCard.value) knowledgeCard.value.status = 'running'
  try {
    await invoke('reprecipitate_document', { documentId, modelProviderId: providerId, modelKey })
  } catch (err) {
    console.warn('Failed to reprecipitate document', err)
  }
}

// Enqueue a document for precipitation. Gated on the setting + a configured chat
// model + an indexed/tree-ready local doc. The backend SHA256-skips unchanged
// docs, so re-enqueuing a done doc is cheap (no LLM).
function scheduleDocumentPrecipitation(doc) {
  if (!knowledgeEnabled.value || !chatModelConfigured.value) return
  if (!doc || doc.id === 'empty') return
  if (doc.source !== 'local' || doc.indexStatus !== 'indexed' || !doc.treeReady) return
  if (precipitationQueued.has(doc.id)) return
  if (knowledgeByDoc[doc.id]?.status === 'running') return
  precipitationQueued.add(doc.id)
  precipitationQueue.push(doc.id)
  pumpPrecipitation()
}

// Free the in-flight slot for `documentId` (if it owns it) and drain the next.
// Single source of truth for slot release (enqueue failure, terminal event, or
// watchdog timeout) so the serial queue can never wedge in one place but not another.
function releasePrecipitationSlot(documentId) {
  if (precipitationActive !== documentId) return
  if (precipitationWatchdog) { window.clearTimeout(precipitationWatchdog); precipitationWatchdog = null }
  precipitationActive = null
  precipitationQueued.delete(documentId)
  pumpPrecipitation()
}

// Run at most one precipitation at a time (serial), draining the queue as each
// completes (via the lumenfolio://knowledge event).
function pumpPrecipitation() {
  if (precipitationActive || !precipitationQueue.length) return
  if (!knowledgeEnabled.value) { precipitationQueue.length = 0; precipitationQueued.clear(); return }
  const documentId = precipitationQueue.shift()
  precipitationActive = documentId
  // Watchdog: if no terminal `lumenfolio://knowledge` event arrives (backend crash
  // before emit, dropped event), free the slot after a timeout so the queue never
  // wedges for the rest of the session.
  if (precipitationWatchdog) window.clearTimeout(precipitationWatchdog)
  precipitationWatchdog = window.setTimeout(() => {
    console.warn('Knowledge precipitation watchdog fired for', documentId)
    releasePrecipitationSlot(documentId)
  }, 120000)
  const { providerId, modelKey } = parseChatModelOptionId(selectedChatModelId.value)
  invoke('enqueue_document_knowledge', { documentId, modelProviderId: providerId, modelKey })
    .catch((err) => {
      console.warn('Failed to enqueue knowledge precipitation', err)
      releasePrecipitationSlot(documentId)
    })
}

function handleKnowledgeEvent(payload) {
  const documentId = payload?.documentId
  if (!documentId) return
  if (payload.status && payload.status !== 'queued') {
    knowledgeByDoc[documentId] = {
      status: payload.status,
      summary: payload.summary || '',
      entities: Number(payload.entities || 0),
      concepts: Number(payload.concepts || 0),
      keywords: Number(payload.keywords || 0),
      error: payload.error || '',
    }
    // Slot freed: this doc finished (done/failed/skipped) — drain the next.
    releasePrecipitationSlot(documentId)
    // If the currently-open document just finished, refresh its card. If ANOTHER
    // doc finished, the open doc's related list may have changed (new shared
    // concepts) — refresh just that, cheaply.
    if (selectedDocId.value === documentId) {
      loadKnowledgeCard(documentId)
    } else if (payload.status === 'done' && selectedDocId.value) {
      loadRelatedDocuments(selectedDocId.value)
    }
  }
}

function handleVisualIndexEvent(payload) {
  const documentId = payload?.documentId
  if (!documentId) return
  const target = allDocs.value.find((item) => item.id === documentId)
  if (!target) return
  target.visualIndexStatus = payload.status || target.visualIndexStatus || 'pending'
  target.visualIndexVersion = Number(payload.version || target.visualIndexVersion || 0)
  target.visualIndexError = payload.error || ''
  // A visual-index failure is recorded on the document but never raises the global
  // red error bar — the text index is what makes the document usable.
  if (['succeeded', 'failed', 'cancelled'].includes(target.visualIndexStatus)) {
    visualIndexRuns.delete(documentId)
  }
}

function handleSelection(selection) {
  activePage.value = selection.page
  activeBlockId.value = selection.blockId || ''
  activeHighlight.value = null
}

function clearPendingSelection() {
  lastSelection.value = null
}

function clearActiveTranslation() {
  activeTranslation.value = null
}

function retryActiveTranslation() {
  const source = activeTranslation.value?.source
  if (!source) return
  handleTranslateSelection(source, { forceRefresh: true })
}

function handleAskSelection(selection) {
  const selected = selection || lastSelection.value
  if (!selected) return
  handleSelection(selected)
  lastSelection.value = selected
  notesDrawerOpen.value = false
  if (rightCollapsed.value) rightCollapsed.value = false
  nextTick(() => {
    chatFocusRequest.value += 1
  })
}

async function handleTranslateSelection(selection, options = {}) {
  const selected = selection || lastSelection.value
  if (!selected) return
  lastSelection.value = null
  activePage.value = selected.page
  activeBlockId.value = selected.blockId || ''
  activeHighlight.value = null
  inlineTranslateOpen.value = false

  const doc = selectedDocument.value
  const translationId = nextLocalId('translation')
  activeTranslation.value = {
    id: translationId,
    status: 'running',
    source: selected,
    targetLang: translationLang.value,
    translatedText: '',
    provider: '',
    cached: false,
    error: '',
  }

  try {
    const result = await invoke('translate_text', {
      input: {
        documentId: doc.id,
        page: selected.page,
        blockId: selected.blockId || '',
        text: selected.text,
        targetLang: translationLang.value,
        sourceVersion: selected.sourceVersion || '',
        forceRefresh: Boolean(options.forceRefresh),
      },
    })
    if (activeTranslation.value?.id !== translationId) return
    activeTranslation.value = {
      ...activeTranslation.value,
      status: 'succeeded',
      translatedText: result.translatedText || '',
      provider: result.provider || '',
      cached: Boolean(result.cached),
      sourceHash: result.sourceHash || '',
    }
  } catch (err) {
    if (activeTranslation.value?.id !== translationId) return
    activeTranslation.value = {
      ...activeTranslation.value,
      status: 'failed',
      error: err?.message || String(err),
    }
  }
}

function handleNoteSelection(selection) {
  openNoteComposer(selection)
}

function handleRealign() {
  const session = activeSession.value
  if (!session) return
  session.messages.push({
    id: nextLocalId('system'),
    sessionId: session.id,
    role: 'assistant',
    content: {
      en: messages.en.realignDone,
      zh: messages.zh.realignDone,
    },
    citations: [],
  })
}

function sanitizeWorkspaceDropPath(rawPath) {
  if (!rawPath || typeof rawPath !== 'string') return ''
  let path = rawPath.trim()
  if (!path) return ''

  if (path.startsWith('file://')) {
    try {
      const url = new URL(path)
      path = decodeURIComponent(url.pathname || '')
      if (url.hostname && !url.pathname) {
        path = ''
      }
      if (/^\/[a-zA-Z]:\//.test(path)) {
        path = path.slice(1)
      }
    } catch {
      path = path.replace(/^file:\/\//i, '')
    }
  }

  path = path.replace(/\\/g, '/')
  try {
    path = decodeURIComponent(path)
  } catch {
    // Keep the original escaped path if it cannot be decoded.
  }
  if (!path) return ''

  const isWindowsDriveRoot = /^[A-Za-z]:\/$/.test(path)
  const withoutTrailingSlash = path.endsWith('/') && path.length > 1 && !isWindowsDriveRoot
    ? path.replace(/\/+$/g, '')
    : path
  return withoutTrailingSlash
}

function normalizeWorkspaceDropPaths(rawPaths) {
  if (!Array.isArray(rawPaths)) return []
  const seen = new Set()
  const normalized = []
  for (const rawPath of rawPaths) {
    const path = sanitizeWorkspaceDropPath(rawPath)
    if (!path || seen.has(path)) continue
    seen.add(path)
    normalized.push(path)
  }
  return normalized
}

function setWorkspaceDropActive(nextState) {
  workspaceDropActive.value = Boolean(nextState)
  if (!workspaceDropActive.value) {
    workspaceDropTargetRootId.value = ''
    workspaceDropTargetCollectionId.value = ''
  }
}

function setWorkspaceDropTargetRootId(rootId) {
  workspaceDropTargetRootId.value = rootId || ''
}

function tauriPayloadCssPoint(payload) {
  const x = Number(payload?.position?.x ?? NaN)
  const y = Number(payload?.position?.y ?? NaN)
  if (!Number.isFinite(x) || !Number.isFinite(y)) return null
  if (typeof window === 'undefined') return null
  const ratio = window.devicePixelRatio || 1
  return { x: x / ratio, y: y / ratio }
}

function getSidebarRect() {
  if (typeof document === 'undefined') return null
  const sidebarEl = document.querySelector('.sidebar')
  if (!sidebarEl) return null
  const rect = sidebarEl.getBoundingClientRect()
  if (!rect || (rect.width === 0 && rect.height === 0)) return null
  return rect
}

function pointInsideRect(point, rect) {
  if (!point || !rect) return false
  return (
    point.x >= rect.left
    && point.x <= rect.right
    && point.y >= rect.top
    && point.y <= rect.bottom
  )
}

// Knowledge-base pivot (C-d): map an OS-file drag point to the collection row
// under it, so dropping files onto a collection imports them into it. Returns
// '__unfiled__' for the Unfiled row, '' when over no row.
function pickTargetCollectionIdForPoint(point) {
  if (!point || typeof document === 'undefined') return ''
  const rows = Array.from(document.querySelectorAll('.collection-row[data-collection-id]'))
    .map((el) => ({
      id: el.getAttribute('data-collection-id') || '',
      rect: el.getBoundingClientRect(),
    }))
    .filter((entry) => entry.id && entry.rect && entry.rect.height > 0)
  if (!rows.length) return ''
  for (const row of rows) {
    if (point.y >= row.rect.top && point.y <= row.rect.bottom) {
      return row.id
    }
  }
  let nearest = rows[0]
  let nearestDistance = Number.POSITIVE_INFINITY
  for (const row of rows) {
    const center = row.rect.top + row.rect.height / 2
    const distance = Math.abs(center - point.y)
    if (distance < nearestDistance) {
      nearestDistance = distance
      nearest = row
    }
  }
  return nearest?.id || ''
}

function applyTauriDragPosition(payload) {
  const point = tauriPayloadCssPoint(payload)
  const sidebarRect = getSidebarRect()
  const insideSidebar = pointInsideRect(point, sidebarRect)
  setWorkspaceDropActive(insideSidebar)
  if (insideSidebar) {
    workspaceDropTargetCollectionId.value = pickTargetCollectionIdForPoint(point)
  } else {
    workspaceDropTargetCollectionId.value = ''
  }
  return { point, insideSidebar }
}

// Resolve the hovered collection row to an import target: '__unfiled__' → null
// (inbox), '' (over no row) → the currently-selected collection.
function resolveDropCollectionId() {
  const hovered = workspaceDropTargetCollectionId.value
  if (hovered === '__unfiled__') return null
  if (hovered) return hovered
  return selectedCollectionId.value ?? null
}

async function addWorkspaceRootsFromDrop(rawPaths, options = {}) {
  if (workspaceStatus.value === 'scanning' || workspaceStatus.value === 'choosing') return false
  const sourcePaths = normalizeWorkspaceDropPaths(rawPaths)
  if (!sourcePaths.length) return false
  // Knowledge-base pivot: imports file into a logical collection (null = unfiled).
  // Prefer an explicit per-call target, else the currently-selected collection.
  const targetCollectionId = options.targetCollectionId ?? selectedCollectionId.value ?? null
  ignoreNextTauriFileDrop.value = true
  lastWorkspaceDropAt.value = Date.now()
  workspaceStatus.value = 'scanning'
  workspaceError.value = ''
  const previousDocId = selectedDocId.value
  try {
    let snapshots = []
    try {
      snapshots = await invoke('import_workspace_paths', {
        args: {
          targetCollectionId,
          sourcePaths,
        },
      })
    } catch (err) {
      workspaceError.value = err?.message || String(err)
      return false
    }
    if (!Array.isArray(snapshots) || !snapshots.length) return false
    snapshots.forEach(upsertWorkspaceRootSnapshot)
    await loadCollections()
    const previousDocStillExists = allDocs.value.some((doc) => doc.id === previousDocId)
    // Keep the prior selection if it survived; otherwise stay on the home
    // surface rather than force-opening an imported document.
    selectedDocId.value = previousDocStillExists ? previousDocId : ''
    if (selectedDocId.value) loadChatHistoryForDocument(selectedDocId.value)
  } finally {
    workspaceStatus.value = 'idle'
    if (fileDropIgnoreTimer) clearTimeout(fileDropIgnoreTimer)
    fileDropIgnoreTimer = setTimeout(() => {
      ignoreNextTauriFileDrop.value = false
    }, WORKSPACE_FILE_DROP_DEBOUNCE_MS)
  }
  return true
}

async function handleWorkspaceDrop(rawPaths = []) {
  const targetCollectionId = resolveDropCollectionId()
  setWorkspaceDropActive(false)
  await addWorkspaceRootsFromDrop(rawPaths, { targetCollectionId })
}

// "+" on a folder header: a discoverable alternative to drag-and-drop. Opens a
// native PDF file picker and adds the chosen files to that folder via the same
// import flow as a drop onto the folder.
// "Import files…" from the library "+" menu: a type-agnostic multi-select picker
// (PDF / Office / Markdown / Text) that funnels through the same import flow as
// drag-and-drop, with no target root (each source lands in a root by its folder).
async function importFilesToLibrary() {
  if (workspaceStatus.value === 'scanning' || workspaceStatus.value === 'choosing') return
  workspaceStatus.value = 'choosing'
  let paths = []
  try {
    paths = await invoke('choose_pdf_files')
  } catch (err) {
    workspaceError.value = err?.message || String(err)
  } finally {
    workspaceStatus.value = 'idle'
  }
  if (Array.isArray(paths) && paths.length) {
    await addWorkspaceRootsFromDrop(paths)
  }
}

async function addPdfsToRoot(rootId) {
  if (workspaceStatus.value === 'scanning' || workspaceStatus.value === 'choosing') return
  workspaceStatus.value = 'choosing'
  let paths = []
  try {
    paths = await invoke('choose_pdf_files')
  } catch (err) {
    workspaceError.value = err?.message || String(err)
  } finally {
    workspaceStatus.value = 'idle'
  }
  if (Array.isArray(paths) && paths.length) {
    // Files import into the active collection (the legacy per-root target no
    // longer exists — organization is by collection now).
    await addWorkspaceRootsFromDrop(paths)
  }
}

async function handleTauriWorkspaceDrop(payload = null, options = {}) {
  const now = Date.now()
  if (workspaceStatus.value === 'scanning' || workspaceStatus.value === 'choosing') return
  let paths = []
  if (Array.isArray(payload?.paths)) {
    paths = payload.paths
  } else if (Array.isArray(payload?.path)) {
    paths = payload.path
  } else if (typeof payload?.path === 'string') {
    paths = [payload.path]
  } else if (Array.isArray(payload)) {
    paths = payload
  } else if (typeof payload === 'string') {
    paths = [payload]
  }
  if (!paths.length) return

  if (ignoreNextTauriFileDrop.value && now - lastWorkspaceDropAt.value < WORKSPACE_FILE_DROP_DEBOUNCE_MS) {
    ignoreNextTauriFileDrop.value = false
    return
  }
  if (ignoreNextTauriFileDrop.value && now - lastWorkspaceDropAt.value >= WORKSPACE_FILE_DROP_DEBOUNCE_MS) {
    ignoreNextTauriFileDrop.value = false
  }
  lastWorkspaceDropAt.value = now
  await addWorkspaceRootsFromDrop(paths, {
    targetCollectionId: options.targetCollectionId ?? null,
  })
}

async function chooseWorkspace() {
  if (workspaceStatus.value === 'scanning' || workspaceStatus.value === 'choosing') return
  workspaceStatus.value = 'choosing'
  workspaceError.value = ''
  let folder = null
  try {
    folder = await invoke('choose_workspace')
  } catch (err) {
    workspaceStatus.value = 'failed'
    workspaceError.value = err?.message || String(err)
    return
  } finally {
    workspaceStatus.value = 'idle'
  }
  if (!folder) return
  // Collections model: import the folder's supported files ONE-SHOT into the
  // active collection (no live disk-scan root, no reconcile binding).
  await addWorkspaceRootsFromDrop([folder])
}

async function rescanWorkspace() {
  if (!workspace.roots.length || workspaceStatus.value === 'choosing' || workspaceStatus.value === 'scanning') {
    return
  }
  try {
    await scanWorkspaces(workspace.roots.map((workspaceRoot) => workspaceRoot.path))
  } catch (err) {
    workspaceStatus.value = 'failed'
    workspaceError.value = err?.message || String(err)
  }
}

async function openWorkspaceInFileManager(rootId = '') {
  const workspaceRoot = workspace.roots.find((item) => item.id === rootId) || activeWorkspaceRoot.value
  const path = String(workspaceRoot?.path || '').trim()
  if (!path) return
  workspaceError.value = ''
  try {
    await invoke('open_path_in_file_manager', { path })
  } catch (err) {
    workspaceError.value = err?.message || String(err)
  }
}

function rootCollapseKey(root) {
  return String(root?.id || root?.path || '')
}

function toggleWorkspaceRoot(rootId) {
  const workspaceRoot = workspace.roots.find((item) => item.id === rootId)
  if (!workspaceRoot) return
  workspaceRoot.collapsed = !workspaceRoot.collapsed
  const key = rootCollapseKey(workspaceRoot)
  if (key) {
    // Replace the whole object: usePersistedRef watches with deep:false, so a
    // nested mutation would not be written back to localStorage.
    collapsedRoots.value = { ...collapsedRoots.value, [key]: workspaceRoot.collapsed }
  }
}

async function reindexSelectedDocument() {
  const doc = selectedDocument.value
  if (!doc || doc.id === 'empty' || workspaceStatus.value === 'scanning') return
  await enqueueBackendDocumentIndex(doc, { force: true })
}

// Reindex a specific document (per-row hover action) without changing the current
// selection. Collapsed-rail reindex still passes no doc and falls back to the
// selected one.
function handleReindexDoc(doc) {
  if (doc && doc.id && doc.id !== 'empty') {
    if (workspaceStatus.value === 'scanning') return
    void enqueueBackendDocumentIndex(doc, { force: true })
    return
  }
  void reindexSelectedDocument()
}

function shouldQueueBackendIndex(doc) {
  if (!doc || doc.id === 'empty' || doc.source !== 'local') return false
  if (doc.backendIndexFailed) return false
  if (['queued', 'indexing'].includes(doc.indexStatus)) return false
  if (doc.indexStatus !== 'indexed') return true
  if (Number(doc.indexVersion || 0) !== Number(doc.currentIndexVersion || 0)) return true
  return !doc.treeReady
}

async function enqueueBackendDocumentIndex(doc, options = {}) {
  if (!doc || doc.id === 'empty') return
  if (!options.force && !shouldQueueBackendIndex(doc)) return
  try {
    workspaceError.value = ''
    doc.indexBeforeReindex = snapshotUsableIndexState(doc)
    resetTranslationAfterIndexChange(doc)
    doc.status = 'stale'
    doc.statusTone = 'warning'
    doc.indexStatus = 'indexing'
    doc.indexVersion = 0
    doc.treeReady = false
    doc.chatReady = false
    doc.indexProgress = indexProgressState(1, 'queued', '')
    doc.backendIndexFailed = false
    doc.translation.error = ''
    const queued = await invoke('enqueue_document_reindex', { documentId: doc.id })
    applyDocumentIndexPendingState(doc, queued)
  } catch (err) {
    doc.backendIndexFailed = true
    workspaceError.value = err?.message || String(err)
    await invoke('mark_document_stale', { documentId: doc.id }).catch(() => {})
    viewerReloadKey.value += 1
  }
}

function handleDocumentIndexEvent(payload) {
  const documentId = payload?.documentId
  if (!documentId) return
  const target = allDocs.value.find((item) => item.id === documentId)
  if (!target) return
  if (payload.status === 'indexed') {
    resetTranslationAfterIndexChange(target)
    delete target.indexBeforeReindex
    target.pageCount = payload.pageCount || target.pageCount
    target.indexVersion = payload.indexVersion || target.currentIndexVersion
    target.status = 'indexed'
    target.statusTone = 'success'
    target.indexStatus = 'indexed'
    target.treeReady = Boolean(payload.treeReady)
    target.chatReady = Boolean(payload.treeReady)
    target.indexProgress = indexProgressState(
      100,
      payload.stage || 'complete',
      payload.stageLabel || '',
    )
    target.visualIndexStatus = payload.visualIndexStatus || 'pending'
    target.visualIndexVersion = Number(payload.visualIndexVersion || 0)
    target.visualIndexError = payload.visualIndexError || ''
    if (selectedDocId.value === documentId) workspaceError.value = ''
    scheduleDocumentVisualIndex(target)
    scheduleDocumentPrecipitation(target)
    return
  }
  if (payload.status === 'failed') {
    if (restoreUsableIndexState(target)) {
      if (selectedDocId.value === documentId) workspaceError.value = ''
      return
    }
    resetTranslationAfterIndexChange(target)
    target.status = 'stale'
    target.statusTone = 'danger'
    target.indexStatus = 'stale'
    target.treeReady = false
    target.chatReady = false
    target.indexProgress = indexProgressState(
      0,
      payload.stage || 'failed',
      payload.stageLabel || '',
    )
    target.backendIndexFailed = true
    // Surface a short, localized message; keep the raw backend error (which can be a
    // multi-line native stack, e.g. a Pdfium dlopen failure) in the console for debugging
    // rather than dumping it into the sidebar.
    if (payload.error) {
      console.warn('Backend document index failed', payload.error)
    }
    workspaceError.value = String(payload.error || '').includes('SCANNED_PDF_NO_TEXT')
      ? ui.value.scannedPdfUnsupported
      : ui.value.backendIndexFallback
    if (selectedDocId.value === documentId) {
      viewerReloadKey.value += 1
    }
    return
  }
  applyDocumentIndexPendingState(target, payload)
}

function snapshotUsableIndexState(doc) {
  if (!doc || doc.indexStatus !== 'indexed') return null
  if (Number(doc.indexVersion || 0) !== Number(doc.currentIndexVersion || 0)) return null
  if (!doc.treeReady) return null
  return {
    status: doc.status,
    statusTone: doc.statusTone,
    indexStatus: doc.indexStatus,
    indexVersion: doc.indexVersion,
    treeReady: doc.treeReady,
    chatReady: doc.chatReady,
    indexProgress: { ...(doc.indexProgress || {}) },
    visualIndexStatus: doc.visualIndexStatus,
    visualIndexVersion: doc.visualIndexVersion,
    visualIndexError: doc.visualIndexError,
  }
}

function restoreUsableIndexState(doc) {
  const previous = doc?.indexBeforeReindex
  delete doc.indexBeforeReindex
  if (!previous) return false
  doc.status = previous.status || 'indexed'
  doc.statusTone = previous.statusTone || 'success'
  doc.indexStatus = previous.indexStatus || 'indexed'
  doc.indexVersion = previous.indexVersion || doc.currentIndexVersion
  doc.treeReady = Boolean(previous.treeReady)
  doc.chatReady = Boolean(previous.chatReady)
  doc.indexProgress = indexProgressState(
    previous.indexProgress?.percent ?? 100,
    previous.indexProgress?.stage || 'complete',
    previous.indexProgress?.label || '',
  )
  doc.visualIndexStatus = previous.visualIndexStatus || doc.visualIndexStatus || 'pending'
  doc.visualIndexVersion = Number(previous.visualIndexVersion || doc.visualIndexVersion || 0)
  doc.visualIndexError = previous.visualIndexError || ''
  doc.backendIndexFailed = true
  return true
}

function applyDocumentIndexPendingState(doc, payload = {}) {
  const event = typeof payload === 'string' ? { status: payload } : (payload || {})
  const wasIndexing = doc.indexStatus === 'indexing'
  if (!wasIndexing) {
    resetTranslationAfterIndexChange(doc)
  }
  const status = event.status
  const normalized = status === 'queued' ? 'indexing' : status || 'indexing'
  const previousPercent = Number(doc.indexProgress?.percent || 0)
  const eventPercent = event.progressPercent == null
    ? previousPercent
    : Number(event.progressPercent)
  const nextPercent = wasIndexing
    ? Math.max(
        Number.isFinite(previousPercent) ? previousPercent : 0,
        Number.isFinite(eventPercent) ? eventPercent : 0,
      )
    : eventPercent
  doc.status = 'indexing'
  doc.statusTone = 'warning'
  doc.indexStatus = normalized
  doc.indexVersion = 0
  doc.treeReady = false
  doc.chatReady = false
  doc.indexProgress = indexProgressState(
    nextPercent,
    event.stage || doc.indexProgress?.stage || normalized,
    event.stageLabel || doc.indexProgress?.label || '',
  )
}

function indexProgressState(percent, stage, label) {
  const numeric = Number(percent)
  return {
    percent: Number.isFinite(numeric) ? Math.max(0, Math.min(100, Math.round(numeric))) : 0,
    stage: String(stage || ''),
    label: String(label || ''),
  }
}

function workspaceRootName(path) {
  const normalized = String(path || '').replace(/[/\\]+$/, '')
  const name = normalized.split(/[/\\]/).filter(Boolean).pop()
  return name || 'Workspace'
}

function createWorkspaceRoot(snapshot) {
  const rootSnapshot = snapshot?.root || {}
  const rootId = String(rootSnapshot.id || '')
  const path = String(rootSnapshot.path || '')
  const existing = workspace.roots.find((item) => item.id === rootId)
  const docs = Array.isArray(snapshot?.documents)
    ? snapshot.documents.map(createLocalDocument)
    : []
  // Mid-session (re-scan) keep the live in-memory state; on a fresh load
  // (no existing root) seed from the persisted collapse map.
  const persistedKey = rootId || path
  const collapsed = existing
    ? Boolean(existing.collapsed)
    : Boolean(persistedKey && collapsedRoots.value?.[persistedKey])
  // Knowledge-base pivot (P2): the virtual root that owns notes/clips gets a
  // friendly localized label instead of its synthetic `lumenfolio://` path.
  const isKnowledgeRoot = rootId === 'root-knowledge-base'
  return {
    id: rootId,
    name: isKnowledgeRoot
      ? { en: 'My knowledge base', zh: '我的知识库' }
      : {
        en: workspaceRootName(path),
        zh: workspaceRootName(path),
      },
    path,
    collapsed,
    folders: [{
      id: `${rootId || path || 'workspace'}-pdfs`,
      name: { en: 'PDFs', zh: 'PDF 文件' },
      docs,
    }],
    recents: docs.slice(0, 5).map((doc) => doc.id),
  }
}

function upsertWorkspaceRootSnapshot(snapshot) {
  const nextRoot = createWorkspaceRoot(snapshot)
  if (!nextRoot.id && !nextRoot.path) return { root: nextRoot, existed: false }
  const existingIndex = workspace.roots.findIndex((item) => (
    item.id === nextRoot.id || (nextRoot.path && item.path === nextRoot.path)
  ))
  const existed = existingIndex >= 0
  if (existingIndex >= 0) {
    workspace.roots.splice(existingIndex, 1, nextRoot)
  } else {
    workspace.roots.push(nextRoot)
  }
  return { root: nextRoot, existed }
}

async function scanWorkspace(folder, options = {}) {
  const previousDocId = selectedDocId.value
  workspaceStatus.value = 'scanning'
  workspaceError.value = ''
  const snapshot = await invoke('scan_workspace_pdfs', { root: folder })
  const { root: workspaceRoot, existed } = upsertWorkspaceRootSnapshot(snapshot)
  const docs = workspaceRoot.folders.flatMap((item) => item.docs)
  const previousDocStillExists = allDocs.value.some((doc) => doc.id === previousDocId)
  const shouldPreserveSelection = options.preserveSelection || options.mode === 'refresh' || existed || !docs.length
  selectedDocId.value = shouldPreserveSelection && previousDocStillExists
    ? previousDocId
    : docs[0]?.id || ''
  if (selectedDocId.value) loadChatHistoryForDocument(selectedDocId.value)
  workspaceStatus.value = 'idle'
}

async function scanWorkspaces(folders) {
  const previousDocId = selectedDocId.value
  workspaceStatus.value = 'scanning'
  workspaceError.value = ''
  const snapshots = []
  for (const folder of folders) {
    // Virtual roots (e.g. the knowledge base at lumenfolio://knowledge) are not
    // filesystem directories — scanning them as folders fails with os error 2.
    // They are managed separately (notes live in the DB), so skip them here.
    if (!folder || String(folder).startsWith('lumenfolio://')) continue
    snapshots.push(await invoke('scan_workspace_pdfs', { root: folder }))
  }
  snapshots.forEach(upsertWorkspaceRootSnapshot)
  const previousDocStillExists = allDocs.value.some((doc) => doc.id === previousDocId)
  selectedDocId.value = previousDocStillExists ? previousDocId : ''
  if (selectedDocId.value) loadChatHistoryForDocument(selectedDocId.value)
  workspaceStatus.value = 'idle'
}

async function loadLastWorkspace() {
  workspaceStatus.value = 'loading'
  workspaceError.value = ''
  try {
    const snapshot = await invoke('load_last_workspace')
    // Load logical collections alongside the doc pool so the sidebar tree can
    // group documents by collection (independent of whether any roots exist).
    await loadCollections()
    if (!Array.isArray(snapshot?.roots) || !snapshot.roots.length) {
      workspaceStatus.value = 'idle'
      return
    }
    workspace.roots = snapshot.roots.map((rootSnapshot) => createWorkspaceRoot(rootSnapshot))
    // Knowledge-base pivot: launch on the "ask my knowledge base" home, not a
    // document. PDFs are no longer the top-priority surface, so we don't
    // auto-open the last-viewed (or first) source — the user opens one on demand.
    selectedDocId.value = ''
    // Restore the tab working set, dropping any docs deleted/moved since last run,
    // and ensure the active doc is always present as a tab.
    const savedTabs = readPersisted('openTabs', [])
    const restoredTabs = (Array.isArray(savedTabs) ? savedTabs : [])
      .filter((id) => allDocs.value.some((doc) => doc.id === id))
    if (selectedDocId.value && !restoredTabs.includes(selectedDocId.value)) {
      restoredTabs.push(selectedDocId.value)
    }
    openTabs.value = restoredTabs
    if (selectedDocId.value) loadChatHistoryForDocument(selectedDocId.value)
    workspaceStatus.value = 'idle'
  } catch (err) {
    workspaceStatus.value = 'failed'
    workspaceError.value = err?.message || String(err)
  }
}

function createLocalDocument(pdf) {
  const title = pdf.title || pdf.short_title || 'Untitled.pdf'
  const indexStatus = pdf.index_status || 'pending'
  const indexVersion = Number(pdf.index_version || 0)
  const currentIndexVersion = Number(pdf.current_index_version || 0)
  const indexFresh = indexStatus === 'indexed' && indexVersion === currentIndexVersion
  return {
    id: pdf.id,
    source: 'local',
    workspaceRootId: pdf.workspace_root_id || '',
    // Knowledge-base pivot (P2): source kind. Editable text sources
    // (note/markdown/text/web) open in the Markdown editor instead of the reader.
    contentType: pdf.content_type || 'pdf',
    // Knowledge-base pivot: the logical collection this source is filed into
    // (null = unfiled). Drives the sidebar collection tree grouping.
    collectionId: pdf.collection_id ?? null,
    // Manual sibling order within the collection; drives drag-reorder in the tree.
    position: Number(pdf.position ?? 0),
    path: pdf.path,
    title,
    shortTitle: pdf.short_title || title,
    status: indexFresh ? 'indexed' : indexStatus === 'stale' ? 'stale' : 'indexing',
    statusTone: indexFresh ? 'success' : indexStatus === 'stale' ? 'danger' : 'warning',
    indexStatus: indexFresh ? 'indexed' : indexStatus,
    indexVersion,
    currentIndexVersion,
    treeReady: indexFresh && Boolean(pdf.tree_ready),
    backendIndexFailed: false,
    visualIndexStatus: pdf.visual_index_status || 'pending',
    visualIndexVersion: Number(pdf.visual_index_version || 0),
    visualIndexError: pdf.visual_index_error || '',
    indexProgress: indexProgressState(
      indexFresh ? 100 : 0,
      indexFresh ? 'complete' : indexStatus,
      '',
    ),
    lastOpened: {
      en: formatFileSize(pdf.size),
      zh: formatFileSize(pdf.size),
    },
    pageCount: pdf.page_count || 0,
    currentPage: pdf.current_page || 1,
    // Don't pin a specific model on a freshly-loaded doc — that hardcoded the first
    // configured model (an HTTP provider, since those precede local agents in the
    // list), so opening a doc clobbered the user's current pick (e.g. Codex) with
    // e.g. deepseek. UNCONFIGURED makes applySelectedChatModel resolve to the global
    // default, which honors the user's last explicit pick (lastChatModelId).
    chatModelId: UNCONFIGURED_CHAT_MODEL_ID,
    quoteBlockId: '',
    chatReady: indexFresh,
    translation: {
      status: 'idle',
      progress: 0,
      total: 0,
      failedBlocks: 0,
      lang: translationLang.value,
      error: '',
      jobId: '',
      providerKey: '',
      phase: '',
      currentPage: 0,
      pdfJobId: '',
      pdfStatus: 'idle',
      pdfProgressPercent: 0,
      monoPdfPath: '',
      dualPdfPath: '',
      pdfArtifactScope: '',
      pdfArtifactPages: '',
      partialArtifacts: {},
      cached: false,
      pages: {},
    },
    pages: [],
    chatHistoryLoaded: false,
    chatHistoryLoading: null,
    messages: [],
    notes: [],
    notesLoaded: false,
    notesLoading: null,
  }
}

function createHighlight(source) {
  return {
    page: source.page,
    bboxList: source.bboxList || [],
  }
}

function formatFileSize(size) {
  if (!Number.isFinite(size) || size <= 0) return ''
  if (size < 1024 * 1024) return `${Math.round(size / 1024)} KB`
  return `${(size / 1024 / 1024).toFixed(1)} MB`
}

function startResize(event) {
  const startX = event.clientX
  const startWidth = rightWidth.value
  const onMove = (moveEvent) => {
    const delta = startX - moveEvent.clientX
    rightWidth.value = Math.max(420, Math.min(680, startWidth + delta))
  }
  const onUp = () => {
    window.removeEventListener('mousemove', onMove)
    window.removeEventListener('mouseup', onUp)
    dragCleanup = null
  }
  window.addEventListener('mousemove', onMove)
  window.addEventListener('mouseup', onUp)
  dragCleanup = onUp
}

function toggleCollapse() {
  rightCollapsed.value = !rightCollapsed.value
}

function toggleLeftCollapse() {
  leftCollapsed.value = !leftCollapsed.value
}

function markStartup(label) {
  const markName = `lumenfolio:${label}`
  performance.mark?.(markName)
  if (label !== 'app-mounted') {
    try {
      performance.measure?.(`lumenfolio:main-to-${label}`, 'lumenfolio:main-start', markName)
    } catch {
      // Performance marks are diagnostic only.
    }
  }
}

function afterFirstPaint(task) {
  window.requestAnimationFrame(() => {
    window.requestAnimationFrame(task)
  })
}

function scheduleIdleTask(task, timeout = 1200) {
  if ('requestIdleCallback' in window) {
    window.requestIdleCallback(task, { timeout })
    return
  }
  window.setTimeout(task, Math.min(timeout, 250))
}

function loadLastWorkspaceAfterFirstPaint() {
  // Capture the saved discovery-view flag BEFORE loadLastWorkspace runs: selecting
  // the restored document inside it flips trendingView off (watch on selectedDocId),
  // and that write would clobber the persisted value before we can restore it.
  const wantTrending = readPersisted('trendingView', false)
  return loadLastWorkspace().finally(() => {
    markStartup('workspace-loaded')
    // Re-open the discovery feed ONLY if the user explicitly left off there. The
    // empty state no longer implies Trending — a fresh/empty library opens to the
    // conversation, so this never fires unless trendingView was truly persisted on.
    if (wantTrending && trendingEnabled.value) trendingView.value = true
    // Fetch the feed only when Trending is the surface the user is actually on.
    // No documents no longer triggers a startup network call — the center is chat.
    if (showTrending.value && trendingStatus.value === 'idle') fetchTrending()
    // Best-effort "newer version?" check — silent on failure, off the critical path.
    checkForUpdate()
    // Knowledge precipitation backfill: enqueue every already-indexed document so
    // the existing library gets sedimented (the serial queue + backend SHA256-skip
    // keep it cheap — done docs cost only a hash compare, no LLM).
    backfillPrecipitation()
  })
}

function backfillPrecipitation() {
  if (!knowledgeEnabled.value || !chatModelConfigured.value) return
  for (const doc of allDocs.value) {
    scheduleDocumentPrecipitation(doc)
  }
}

async function probePdfTranslationRuntime() {
  try {
    const result = await invoke('probe_pdf_translation_runtime')
    pdfTranslationRuntime.value = {
      checked: true,
      ok: Boolean(result?.ok),
      error: result?.error || '',
    }
  } catch (err) {
    pdfTranslationRuntime.value = {
      checked: true,
      ok: false,
      error: err?.message || String(err),
    }
  }
}

onBeforeUnmount(() => {
  clearInterval(translationTimer)
  clearTimeout(translationPageRefreshTimer)
  cancelQueuedTranslationPageLoads()
  if (assistantStreamDrainTimer) clearTimeout(assistantStreamDrainTimer)
  if (dragCleanup) dragCleanup()
  if (agentActivityUnlisten) agentActivityUnlisten()
  if (answerDeltaUnlisten) answerDeltaUnlisten()
  if (reasoningDeltaUnlisten) reasoningDeltaUnlisten()
  if (askDocumentDoneUnlisten) askDocumentDoneUnlisten()
  if (askDocumentErrorUnlisten) askDocumentErrorUnlisten()
  if (askDocumentStoppedUnlisten) askDocumentStoppedUnlisten()
  if (documentIndexUnlisten) documentIndexUnlisten()
  if (visualIndexUnlisten) visualIndexUnlisten()
  if (knowledgeUnlisten) knowledgeUnlisten()
  if (translationJobUnlisten) translationJobUnlisten()
  if (pdfTranslationUnlisten) pdfTranslationUnlisten()
  if (dragEnterUnlisten) dragEnterUnlisten()
  if (dragOverUnlisten) dragOverUnlisten()
  if (dragLeaveUnlisten) dragLeaveUnlisten()
  if (dragDropUnlisten) dragDropUnlisten()
  if (fileDropIgnoreTimer) clearTimeout(fileDropIgnoreTimer)
  window.removeEventListener('keydown', handleGlobalKeydown)
})

// ESC exits the full-screen global knowledge graph back to the reader. Guarded
// to graph-open so it never swallows ESC from inputs/other views.
function handleGlobalKeydown(event) {
  if ((event.metaKey || event.ctrlKey) && !event.altKey && event.key.toLowerCase() === 'k') {
    event.preventDefault()
    paletteOpen.value = !paletteOpen.value
    return
  }
  if (event.key === 'Escape' && graphView.value) {
    handleCloseGraph()
  }
}

onMounted(() => {
  // TEMP: report any main-thread long task (>200ms). If pdf.js used a real
  // worker, font work would NOT appear here — so a long task during translate
  // means the heavy work is on the main thread. Remove after diagnosis.
  try {
    const lto = new PerformanceObserver((list) => {
      for (const entry of list.getEntries()) {
        if (entry.duration < 200) continue
        const attr = (entry.attribution && entry.attribution[0]) || {}
        // eslint-disable-next-line no-console
        console.log(`[longtask] ${Math.round(entry.duration)}ms container=${attr.containerType || '?'}/${attr.containerName || ''} src=${attr.containerSrc || ''}`)
      }
    })
    lto.observe({ type: 'longtask', buffered: true })
  } catch (err) {
    // longtask API unavailable — ignore
  }
  window.addEventListener('keydown', handleGlobalKeydown)
  markStartup('app-mounted')
  afterFirstPaint(() => {
    markStartup('app-first-frame')
    loadTranslationSettings()
    loadLastWorkspaceAfterFirstPaint()
    loadSessionList()
    scheduleIdleTask(() => loadModelProviders(), 1200)
    // If the persisted pick is a local agent, detect sooner so the model selector
    // restores it with minimal flash; otherwise detect on the usual idle cadence.
    const localPickPending = String(lastChatModelId.value || '').startsWith('local-agent-')
    scheduleIdleTask(() => refreshLocalAgentStatus(), localPickPending ? 200 : 1600)
    scheduleIdleTask(() => probePdfTranslationRuntime(), 2000)
  })
  listen('lumenfolio://agent-activity', (event) => {
    handleAgentActivity(event.payload)
  }).then((unlisten) => {
    agentActivityUnlisten = unlisten
  }).catch((err) => {
    console.warn('Failed to listen for agent activity', err)
  })
  listen('lumenfolio://answer-delta', (event) => {
    handleAnswerDelta(event.payload)
  }).then((unlisten) => {
    chatStreamDebug('listener ready', { event: 'lumenfolio://answer-delta' })
    answerDeltaUnlisten = unlisten
  }).catch((err) => {
    console.warn('Failed to listen for answer stream', err)
  })
  listen('lumenfolio://reasoning-delta', (event) => {
    handleReasoningDelta(event.payload)
  }).then((unlisten) => {
    reasoningDeltaUnlisten = unlisten
  }).catch((err) => {
    console.warn('Failed to listen for reasoning stream', err)
  })
  listen('lumenfolio://ask-document-done', (event) => {
    applyAskDocumentResult(event.payload?.eventId, event.payload?.result)
  }).then((unlisten) => {
    chatStreamDebug('listener ready', { event: 'lumenfolio://ask-document-done' })
    askDocumentDoneUnlisten = unlisten
  }).catch((err) => {
    console.warn('Failed to listen for ask completion', err)
  })
  listen('lumenfolio://ask-document-error', (event) => {
    applyAskDocumentError(event.payload?.eventId, event.payload?.message)
  }).then((unlisten) => {
    askDocumentErrorUnlisten = unlisten
  }).catch((err) => {
    console.warn('Failed to listen for ask errors', err)
  })
  listen('lumenfolio://ask-document-stopped', (event) => {
    applyAskDocumentStopped(event.payload?.eventId)
  }).then((unlisten) => {
    askDocumentStoppedUnlisten = unlisten
  }).catch((err) => {
    console.warn('Failed to listen for ask stop', err)
  })
  listen('lumenfolio://document-index', (event) => {
    handleDocumentIndexEvent(event.payload)
  }).then((unlisten) => {
    documentIndexUnlisten = unlisten
  }).catch((err) => {
    console.warn('Failed to listen for document index events', err)
  })
  listen('lumenfolio://visual-index', (event) => {
    handleVisualIndexEvent(event.payload)
  }).then((unlisten) => {
    visualIndexUnlisten = unlisten
  }).catch((err) => {
    console.warn('Failed to listen for visual index events', err)
  })
  listen('lumenfolio://knowledge', (event) => {
    handleKnowledgeEvent(event.payload)
  }).then((unlisten) => {
    knowledgeUnlisten = unlisten
  }).catch((err) => {
    console.warn('Failed to listen for knowledge events', err)
  })
  listen('lumenfolio://translation-job', (event) => {
    handleTranslationJobEvent(event.payload)
  }).then((unlisten) => {
    translationJobUnlisten = unlisten
  }).catch((err) => {
    console.warn('Failed to listen for translation job events', err)
  })
  listen('lumenfolio://pdf-translation', (event) => {
    handlePdfTranslationEvent(event.payload)
  }).then((unlisten) => {
    pdfTranslationUnlisten = unlisten
  }).catch((err) => {
    console.warn('Failed to listen for PDF translation events', err)
  })
  listen('tauri://drag-enter', (event) => {
    applyTauriDragPosition(event.payload)
  }).then((unlisten) => {
    dragEnterUnlisten = unlisten
  }).catch((err) => {
    console.warn('Failed to listen for drag enter events', err)
  })
  listen('tauri://drag-over', (event) => {
    applyTauriDragPosition(event.payload)
  }).then((unlisten) => {
    dragOverUnlisten = unlisten
  }).catch((err) => {
    console.warn('Failed to listen for drag over events', err)
  })
  listen('tauri://drag-leave', () => {
    setWorkspaceDropActive(false)
  }).then((unlisten) => {
    dragLeaveUnlisten = unlisten
  }).catch((err) => {
    console.warn('Failed to listen for drag leave events', err)
  })
  listen('tauri://drag-drop', (event) => {
    const payload = event.payload
    const { insideSidebar } = applyTauriDragPosition(payload)
    // Import into the collection under the cursor (falls back to the selected
    // collection, or the inbox when dropped on Unfiled).
    const targetCollectionId = resolveDropCollectionId()
    setWorkspaceDropActive(false)
    if (!insideSidebar) return
    void handleTauriWorkspaceDrop(payload, { targetCollectionId })
  }).then((unlisten) => {
    dragDropUnlisten = unlisten
  }).catch((err) => {
    console.warn('Failed to listen for drag drop events', err)
  })
})
</script>

<template>
  <div class="app-shell" :class="{ 'left-collapsed': leftCollapsed }">
    <div v-if="showUpdateBanner" class="update-banner" role="status">
      <span class="update-banner-icon" aria-hidden="true">⬆</span>
      <span class="update-banner-text">{{ ui.updateAvailable }} v{{ updateInfo.latest }}</span>
      <button type="button" class="update-banner-view" @click="openUpdatePage">{{ ui.updateView }}</button>
      <button
        type="button"
        class="update-banner-dismiss"
        :aria-label="ui.dismiss || 'Dismiss'"
        @click="dismissUpdateBanner"
      >×</button>
    </div>
    <div v-if="transientToast" class="update-banner" role="status">
      <span class="update-banner-text">{{ transientToast }}</span>
    </div>
    <WorkspaceSidebar
      :roots="workspace.roots"
      :collections="collections"
      :documents="allDocs"
      :home-active="conversationCentered"
      :selected-doc-id="selectedDocId"
      :selected-doc="selectedDocument"
      :filter="filter"
      :scan-status="workspaceStatus"
      :scan-error="workspaceError"
      :collapsed="leftCollapsed"
      :locale="locale"
      :ui="ui"
      :drop-active="workspaceDropActive"
      :drop-target-root-id="workspaceDropTargetRootId"
      :drop-target-collection-id="workspaceDropTargetCollectionId"
      :trending-active="showTrending"
      :trending-enabled="trendingEnabled"
      :graph-active="showGraph"
      :graph-enabled="knowledgeEnabled"
      @update:filter="filter = $event"
      @open-trending="handleOpenTrending"
      @open-graph="handleOpenGraph"
      @new-note="handleCreateNote"
      @select-doc="selectDoc"
      @add-folder="chooseWorkspace"
      @import-files="importFilesToLibrary"
      @add-pdfs="addPdfsToRoot"
      @rescan="rescanWorkspace"
      @reindex-doc="handleReindexDoc"
      @reveal-doc="handleRevealDoc"
      @delete-doc="openDeleteDocumentConfirm"
      @open-workspace="openWorkspaceInFileManager"
      @delete-root="openRemoveWorkspaceRootConfirm"
      @set-drop-active="setWorkspaceDropActive"
      @workspace-drop="handleWorkspaceDrop"
      @toggle-root="toggleWorkspaceRoot"
      @open-settings="openSettings"
      @toggle-collapse="toggleLeftCollapse"
      @new-collection="handleCreateCollection"
      @rename-collection="handleRenameCollection"
      @delete-collection="handleDeleteCollection"
      @select-collection="handleSelectCollection"
      @move-doc-to-collection="handleMoveDocToCollection"
      @move-collection="handleMoveCollection"
      @reorder-documents="handleReorderDocuments"
      @reorder-collections="handleReorderCollections"
      @clear-unfiled="handleClearUnfiled"
      @go-home="goHome"
    />

    <button
      type="button"
      class="left-collapse-btn"
      :class="{ collapsed: leftCollapsed }"
      :title="leftCollapsed ? ui.expand : ui.collapse"
      :aria-label="leftCollapsed ? ui.expand : ui.collapse"
      @click="toggleLeftCollapse"
    >
      {{ leftCollapsed ? '❯' : '❮' }}
    </button>

    <TrendingFeed
      v-if="showTrending"
      :papers="trendingPapers"
      :status="trendingStatus"
      :error="trendingError"
      :add-error="trendingAddError"
      :adding-ids="trendingAddingIds"
      :ui="ui"
      :locale="locale"
      :period="trendingPeriod"
      @refresh="fetchTrending"
      @set-period="handleTrendingPeriod"
      @add-paper="handleAddTrendingPaper"
      @open-hf="handleOpenHf"
    />

    <KnowledgeGraphView
      v-if="showGraph"
      :data="graphData"
      :status="graphStatus"
      :error="graphError"
      :selected-doc-id="selectedDocId"
      :initial-focus="graphFocusOnOpen"
      :ui="ui"
      @open-doc="selectDoc"
      @refresh="fetchKnowledgeGraph"
      @close="handleCloseGraph"
    />

    <div v-if="!showTrending && !showGraph && !conversationCentered && officeSelected" class="reader-column">
      <OfficeViewer
        :key="selectedDocument.id"
        :document-id="selectedDocument.id"
        :content-type="selectedDocument.contentType"
        :title="selectedDocument.shortTitle || selectedDocument.title || ''"
        :citation="activeCitation"
        :ui="ui"
      />
    </div>

    <div v-else-if="!showTrending && !showGraph && !conversationCentered && editableSelected" class="reader-column">
      <NoteEditor
        ref="noteEditorRef"
        :key="selectedDocument.id"
        :document-id="selectedDocument.id"
        :index-status="selectedDocument.indexStatus"
        :ui="ui"
        @saved="handleNoteSaved"
        @open-doc="selectDoc"
        @create-from-link="handleCreateNote"
      />
    </div>

    <div v-else-if="!showTrending && !showGraph && !conversationCentered" class="reader-column">
    <KeepAlive>
      <ReaderPane
      :key="`${selectedDocument.id}:${viewerReloadKey}`"
      :document="selectedDocument"
      :translation-languages="translationLanguages"
      :translation-lang="translationLang"
      :view-mode="viewMode"
      :active-page="activePage"
      :active-block-id="activeBlockId"
      :active-highlight="activeHighlight"
      :note-highlights="notesDrawerOpen ? (selectedDocument.notes || []) : []"
      :hovered-linked-block="hoveredLinkedBlock"
      :active-translation="activeTranslation"
      :page-translation="selectedDocument.translation.pages?.[activePage] || null"
      :selection-locked="Boolean(lastSelection)"
      :inline-translate-open="inlineTranslateOpen"
      :locale="locale"
      :knowledge-tab="knowledgeEnabled && selectedDocument.id !== 'empty'"
      :knowledge-active="knowledgeView"
      :knowledge-status="knowledgeByDoc[selectedDocument.id]?.status || knowledgeCard?.status || ''"
      :ui="ui"
      @update:translationLang="translationLang = $event"
      @translation-action="handleTranslationAction"
      @cancel-translation="cancelTranslation"
      @set-view-mode="setViewMode"
      @toggle-inline-translate="toggleInlineTranslate"
      @select-page="setPage"
      @linked-block-hover="setLinkedBlockHover"
      @request-translation-page="handleTranslationPageRequest"
      @request-pdf-translation-pages="handlePdfTranslationPagesRequest"
      @document-loaded="handleDocumentLoaded"
      @document-load-failed="handleDocumentLoadFailed"
      @document-index-started="handleDocumentIndexStarted"
      @document-index-complete="handleDocumentIndexComplete"
      @document-index-failed="handleDocumentIndexFailed"
      @selection="handleSelection"
      @ask-selection="handleAskSelection"
      @translate-selection="handleTranslateSelection"
      @note-selection="handleNoteSelection"
      @close-translation="clearActiveTranslation"
      @retry-translation="retryActiveTranslation"
      @realign="handleRealign"
      @set-knowledge-view="knowledgeView = $event"
    >
      <template #overlay>
        <KnowledgePane
          v-if="knowledgeEnabled && knowledgeView && selectedDocument.id !== 'empty'"
          :card="knowledgeCard"
          :related="relatedDocs"
          :related-links="relatedLinks"
          :document-title="selectedDocument.title || ''"
          :live-status="knowledgeByDoc[selectedDocument.id]?.status || ''"
          :ui="ui"
          @reprecipitate="reprecipitateCurrentDocument"
          @open-doc="selectDoc"
          @open-graph="handleOpenGraphFocused"
        />
      </template>
    </ReaderPane>
    </KeepAlive>
    </div>

    <div v-if="!rightCollapsed && !conversationCentered" class="drag-handle" @mousedown.prevent="startResize" />

    <ChatPane
      :session="activeSession"
      :document="activeFocusDoc"
      :viewed-doc-id="selectedDocId"
      :viewed-doc-name="selectedDocument.shortTitle || selectedDocument.title || ''"
      :browsing-label="chatBrowsingLabel"
      :running-event-id="globalRunningEventId"
      :all-documents="allDocs"
      :centered="conversationCentered"
      :collapsed="!conversationCentered && rightCollapsed"
      :clipping="clipBusy"
      :graph-enabled="knowledgeEnabled"
      :width="rightWidth"
      :active-citation-id="activeCitationId"
      :available-models="availableChatModels"
      :current-model-id="selectedChatModelId"
      :current-model="currentChatModel"
      :model-configured="chatModelConfigured"
      :pending-selection="lastSelection"
      :focus-request="chatFocusRequest"
      :suggested-questions="suggestedQuestions"
      :web-search-enabled="webSearchEnabled"
      :sessions="sessionTabs"
      :history-items="sessionHistoryItems"
      :history-open="sessionHistoryOpen"
      :notes-open="notesDrawerOpen"
      :locale="locale"
      :ui="ui"
      @toggle-collapse="toggleCollapse"
      @citation-click="handleCitationClick"
      @update:model-id="handleSelectChatModel"
      @clear-selection="clearPendingSelection"
      @clear-history="openClearChatHistoryConfirm"
      @new-session="handleNewSession"
      @export-chat="handleExportChat"
      @toggle-web-search="webSearchEnabled = !webSearchEnabled"
      @stop-generation="handleStopGeneration"
      @select-session="setActiveSession"
      @close-session="closeSessionTab"
      @delete-session="deleteSessionById"
      @toggle-history="sessionHistoryOpen = !sessionHistoryOpen"
      @close-history="sessionHistoryOpen = false"
      @toggle-notes="toggleNotesDrawer"
      @set-focus-doc="handleSetSessionFocus"
      @edit-resend="handleEditResend"
      @send="handleSend"
      @open-doc="selectDoc"
      @new-note="handleCreateNote"
      @clip-web="handleClipWebPage"
      @open-graph="handleOpenGraph"
      @apply-note-edit="handleApplyNoteEdit"
    />

    <Transition name="notes-drawer">
      <div v-if="notesDrawerOpen && !rightCollapsed" class="notes-drawer">
        <NotesPane
          :document="selectedDocument"
          :collapsed="false"
          :width="330"
          :notes="selectedDocument.notes || []"
          :loading="Boolean(selectedDocument.notesLoading) && !selectedDocument.notesLoaded"
          :active-note-id="activeNoteId"
          :as-drawer="true"
          :locale="locale"
          :ui="ui"
          @toggle-collapse="toggleNotesDrawer"
          @set-tab="setRightPaneTab"
          @note-focus="focusNote"
          @note-edit="openNoteEditComposer"
          @note-delete="openNoteDeleteConfirm"
        />
      </div>
    </Transition>

    <NoteComposer
      :show="noteComposer.open"
      :mode="noteComposer.mode"
      :quote-text="noteComposer.quoteText"
      :initial-content="noteComposer.content"
      :saving="noteComposerSaving"
      :ui="ui"
      @save="submitNoteComposer"
      @cancel="closeNoteComposer"
    />

    <div v-if="noteDeleteConfirmOpen" class="confirm-backdrop" @click.self="closeNoteDeleteConfirm">
      <section class="confirm-modal" role="dialog" aria-modal="true" :aria-label="ui.noteDeleteTitle">
        <div class="confirm-head">
          <div class="confirm-title">{{ ui.noteDeleteTitle }}</div>
          <button
            class="confirm-close"
            type="button"
            :aria-label="ui.close"
            :disabled="noteDeleteStatus === 'deleting'"
            @click="closeNoteDeleteConfirm"
          >
            ×
          </button>
        </div>
        <div class="confirm-body">
          <p>{{ ui.noteDeleteConfirm }}</p>
          <div v-if="noteDeleteTargetPreview" class="confirm-target">{{ noteDeleteTargetPreview }}</div>
          <div v-if="noteDeleteStatus === 'failed'" class="confirm-error">
            {{ ui.noteDeleteFailed }}: {{ noteDeleteError }}
          </div>
        </div>
        <div class="confirm-actions">
          <button
            type="button"
            class="confirm-btn"
            :disabled="noteDeleteStatus === 'deleting'"
            @click="closeNoteDeleteConfirm"
          >
            {{ ui.cancel }}
          </button>
          <button
            type="button"
            class="confirm-btn danger"
            :disabled="noteDeleteStatus === 'deleting'"
            @click="confirmDeleteNote"
          >
            {{ noteDeleteStatus === 'deleting' ? `${ui.delete}...` : ui.delete }}
          </button>
        </div>
      </section>
    </div>

    <div v-if="documentDeleteConfirmOpen" class="confirm-backdrop" @click.self="closeDeleteDocumentConfirm">
      <section class="confirm-modal" role="dialog" aria-modal="true" :aria-label="ui.deleteDocumentTitle">
        <div class="confirm-head">
          <div class="confirm-title">{{ ui.deleteDocumentTitle }}</div>
          <button
            class="confirm-close"
            type="button"
            :aria-label="ui.close"
            :disabled="documentDeleteStatus === 'deleting'"
            @click="closeDeleteDocumentConfirm"
          >
            ×
          </button>
        </div>
        <div class="confirm-body">
          <p>{{ documentDeleteTarget.isNote ? ui.deleteNoteConfirm : ui.deleteDocumentConfirm }}</p>
          <div v-if="documentDeleteTarget.title" class="confirm-target">{{ documentDeleteTarget.title }}</div>
          <div v-if="documentDeleteStatus === 'failed'" class="confirm-error">
            {{ ui.deleteDocumentFailed }}: {{ documentDeleteError }}
          </div>
        </div>
        <div class="confirm-actions">
          <button
            type="button"
            class="confirm-btn"
            :disabled="documentDeleteStatus === 'deleting'"
            @click="closeDeleteDocumentConfirm"
          >
            {{ ui.cancel }}
          </button>
          <button
            type="button"
            class="confirm-btn danger"
            :disabled="documentDeleteStatus === 'deleting'"
            @click="confirmDeleteDocument"
          >
            {{ documentDeleteStatus === 'deleting' ? `${ui.delete}...` : ui.delete }}
          </button>
        </div>
      </section>
    </div>

    <div v-if="providerDeleteConfirmOpen" class="confirm-backdrop" @click.self="closeDeleteProviderConfirm">
      <section class="confirm-modal" role="dialog" aria-modal="true" :aria-label="ui.deleteProviderTitle">
        <div class="confirm-head">
          <div class="confirm-title">{{ ui.deleteProviderTitle }}</div>
          <button
            class="confirm-close"
            type="button"
            :aria-label="ui.close"
            :disabled="providerDeleteStatus === 'deleting'"
            @click="closeDeleteProviderConfirm"
          >
            ×
          </button>
        </div>
        <div class="confirm-body">
          <p>{{ ui.deleteProviderConfirm.replace('{name}', providerDeleteTarget.name) }}</p>
          <div v-if="providerDeleteStatus === 'failed'" class="confirm-error">
            {{ ui.deleteProviderFailed }}: {{ providerDeleteError }}
          </div>
        </div>
        <div class="confirm-actions">
          <button
            type="button"
            class="confirm-btn"
            :disabled="providerDeleteStatus === 'deleting'"
            @click="closeDeleteProviderConfirm"
          >
            {{ ui.cancel }}
          </button>
          <button
            type="button"
            class="confirm-btn danger"
            :disabled="providerDeleteStatus === 'deleting'"
            @click="confirmDeleteProvider"
          >
            {{ providerDeleteStatus === 'deleting' ? `${ui.delete}...` : ui.delete }}
          </button>
        </div>
      </section>
    </div>

    <div v-if="clearChatConfirmOpen" class="confirm-backdrop" @click.self="closeClearChatHistoryConfirm">
      <section class="confirm-modal" role="dialog" aria-modal="true" :aria-label="ui.clearChatHistory">
        <div class="confirm-head">
          <div class="confirm-title">{{ ui.clearChatHistory }}</div>
          <button
            class="confirm-close"
            type="button"
            :aria-label="ui.close"
            :disabled="clearChatStatus === 'clearing'"
            @click="closeClearChatHistoryConfirm"
          >
            ×
          </button>
        </div>
        <div class="confirm-body">
          <p>{{ ui.clearChatHistoryConfirm }}</p>
          <div class="confirm-target">{{ clearChatTargetTitle }}</div>
          <div v-if="clearChatStatus === 'failed'" class="confirm-error">
            {{ ui.clearChatHistoryFailed }}: {{ clearChatError }}
          </div>
        </div>
        <div class="confirm-actions">
          <button
            type="button"
            class="confirm-btn"
            :disabled="clearChatStatus === 'clearing'"
            @click="closeClearChatHistoryConfirm"
          >
            {{ ui.cancel }}
          </button>
          <button
            type="button"
            class="confirm-btn danger"
            :disabled="clearChatStatus === 'clearing'"
            @click="confirmClearChatHistory"
          >
            {{ clearChatStatus === 'clearing' ? `${ui.clearChatHistoryShort}...` : ui.clearChatHistoryShort }}
          </button>
        </div>
      </section>
    </div>

    <div v-if="removeWorkspaceRootConfirmOpen" class="confirm-backdrop" @click.self="closeRemoveWorkspaceRootConfirm">
      <section class="confirm-modal" role="dialog" aria-modal="true" :aria-label="ui.removeWorkspace">
        <div class="confirm-head">
          <div class="confirm-title">{{ ui.removeWorkspace }}</div>
          <button
            class="confirm-close"
            type="button"
            :aria-label="ui.close"
            :disabled="removeWorkspaceRootStatus === 'removing'"
            @click="closeRemoveWorkspaceRootConfirm"
          >
            ×
          </button>
        </div>
        <div class="confirm-body">
          <p>{{ ui.removeWorkspaceConfirm.replace('{name}', removeWorkspaceRootTarget.name || removeWorkspaceRootTarget.path) }}</p>
          <div class="confirm-target">{{ removeWorkspaceRootTarget.path }}</div>
          <p v-if="removeWorkspaceRootTarget.docCount">
            {{ ui.removeWorkspaceSummary }}: {{ removeWorkspaceRootTarget.docCount }}
          </p>
          <div v-if="removeWorkspaceRootStatus === 'failed'" class="confirm-error">
            {{ ui.removeWorkspaceFailed }}: {{ removeWorkspaceRootError }}
          </div>
        </div>
        <div class="confirm-actions">
          <button
            type="button"
            class="confirm-btn"
            :disabled="removeWorkspaceRootStatus === 'removing'"
            @click="closeRemoveWorkspaceRootConfirm"
          >
            {{ ui.cancel }}
          </button>
          <button
            type="button"
            class="confirm-btn danger"
            :disabled="removeWorkspaceRootStatus === 'removing'"
            @click="confirmRemoveWorkspaceRoot"
          >
            {{ removeWorkspaceRootStatus === 'removing' ? `${ui.removeWorkspace}...` : ui.removeWorkspace }}
          </button>
        </div>
      </section>
    </div>

    <CommandPalette
      v-if="paletteOpen"
      :open="paletteOpen"
      :documents="allDocs"
      :actions="paletteActions"
      :ui="ui"
      @close="paletteOpen = false"
      @run-action="runPaletteAction"
      @open-doc="selectDoc"
    />

    <div v-if="settingsOpen" class="settings-backdrop" @click.self="closeSettings">
      <section class="settings-modal" role="dialog" aria-modal="true" :aria-label="ui.modelSettings">
        <div class="settings-head">
          <div>
            <div class="settings-title">{{ ui.modelSettings }}</div>
            <div class="settings-subtitle">{{ ui.providerConfig }}</div>
          </div>
          <button class="settings-close" type="button" :aria-label="ui.close" @click="closeSettings">×</button>
        </div>

        <div class="settings-layout">
          <nav class="settings-nav" :aria-label="ui.settingsSections">
            <button
              type="button"
              class="settings-nav-item"
              :class="{ active: settingsSection === 'general' }"
              @click="switchSettingsSection('general')"
            >
              <span>{{ ui.generalNav }}</span>
              <small>{{ ui.generalNavHint }}</small>
            </button>
            <button
              type="button"
              class="settings-nav-item"
              :class="{ active: settingsSection === 'chat' }"
              @click="switchSettingsSection('chat')"
            >
              <span>{{ ui.chatProvidersNav }}</span>
              <small>{{ ui.chatProvidersNavHint }}</small>
            </button>
            <button
              type="button"
              class="settings-nav-item"
              :class="{ active: settingsSection === 'websearch' }"
              @click="switchSettingsSection('websearch')"
            >
              <span>{{ ui.webSearchNav }}</span>
              <small>{{ ui.webSearchNavHint }}</small>
            </button>
            <button
              type="button"
              class="settings-nav-item"
              :class="{ active: settingsSection === 'translation' }"
              @click="switchSettingsSection('translation')"
            >
              <span>{{ ui.translationNav }}</span>
              <small>{{ ui.translationNavHint }}</small>
            </button>
          </nav>

          <div v-if="settingsSection === 'general'" class="settings-panel settings-body">
            <div class="settings-section-title full">{{ ui.generalNav }}</div>

            <!-- A segmented control rather than a <select>: there are exactly
                 three choices, all worth seeing at once, and appearance is the
                 one setting where the result is visible the instant you pick. -->
            <div class="settings-field full">
              <span>{{ ui.appearance }}</span>
              <div class="theme-choice" role="radiogroup" :aria-label="ui.appearance">
                <button
                  v-for="option in themeChoices"
                  :key="option.id"
                  type="button"
                  role="radio"
                  class="theme-choice-btn"
                  :class="{ active: themePreference === option.id }"
                  :aria-checked="themePreference === option.id"
                  @click="setThemePreference(option.id)"
                >
                  <svg viewBox="0 0 16 16" fill="none" aria-hidden="true">
                    <template v-if="option.id === 'dark'">
                      <path
                        d="M13 9.6A5.4 5.4 0 0 1 6.4 3a5.5 5.5 0 1 0 6.6 6.6Z"
                        stroke="currentColor"
                        stroke-width="1.4"
                        stroke-linejoin="round"
                      />
                    </template>
                    <template v-else-if="option.id === 'light'">
                      <circle cx="8" cy="8" r="3" stroke="currentColor" stroke-width="1.4" />
                      <path
                        d="M8 1.5v1.6M8 12.9v1.6M1.5 8h1.6M12.9 8h1.6M3.4 3.4l1.1 1.1M11.5 11.5l1.1 1.1M12.6 3.4l-1.1 1.1M4.5 11.5l-1.1 1.1"
                        stroke="currentColor"
                        stroke-width="1.4"
                        stroke-linecap="round"
                      />
                    </template>
                    <template v-else>
                      <rect
                        x="1.8"
                        y="3"
                        width="12.4"
                        height="8.4"
                        rx="1.4"
                        stroke="currentColor"
                        stroke-width="1.4"
                      />
                      <path d="M5.5 13.6h5" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" />
                    </template>
                  </svg>
                  {{ option.label }}
                </button>
              </div>
            </div>
            <div class="settings-note full">{{ ui.appearanceHint }}</div>

            <label class="settings-field full">
              <span>{{ ui.interfaceLanguage }}</span>
              <select v-model="locale">
                <option value="en">{{ ui.languageNameEnglish }}</option>
                <option value="zh">{{ ui.languageNameChinese }}</option>
              </select>
            </label>
            <div class="settings-note full">{{ ui.interfaceLanguageHint }}</div>

            <label class="settings-field full settings-toggle">
              <input v-model="trendingEnabled" type="checkbox" />
              <span>{{ ui.trendingDiscovery }}</span>
            </label>
            <div class="settings-note full">{{ ui.trendingDiscoveryHint }}</div>

            <label class="settings-field full settings-toggle">
              <input v-model="knowledgeEnabled" type="checkbox" />
              <span>{{ ui.knowledgePrecipitation }}</span>
            </label>
            <div class="settings-note full">{{ ui.knowledgePrecipitationHint }}</div>

            <!-- Notes are mirrored to plain .md files so they survive the
                 database and so any sync client can carry them. -->
            <div class="settings-field full">
              <span class="settings-label">{{ ui.notesVault }}</span>
              <div class="vault-row">
                <input
                  v-model="vaultDir"
                  type="text"
                  class="vault-input"
                  :placeholder="ui.notesVaultPlaceholder"
                  @change="saveVaultDir"
                />
                <button type="button" class="settings-btn" :disabled="vaultBusy" @click="chooseVaultDir">
                  {{ ui.browse }}
                </button>
              </div>
            </div>
            <div class="settings-note full">{{ ui.notesVaultHint }}</div>
            <div class="settings-field full">
              <div class="vault-row">
                <button type="button" class="settings-btn" :disabled="vaultBusy" @click="exportVault">
                  {{ ui.notesVaultExport }}
                </button>
                <button type="button" class="settings-btn" :disabled="vaultBusy" @click="importVault">
                  {{ ui.notesVaultImport }}
                </button>
                <span v-if="vaultStatus" class="vault-status">{{ vaultStatus }}</span>
              </div>
            </div>

            <!-- Collections and chat history live only in the database; notes are
                 already safe as .md. This snapshots the rest. -->
            <div class="settings-field full">
              <span class="settings-label">{{ ui.backupFolder }}</span>
              <div class="vault-row">
                <input
                  v-model="backupDir"
                  type="text"
                  class="vault-input"
                  :placeholder="ui.backupFolderPlaceholder"
                  @change="saveBackupSettings"
                />
                <button type="button" class="settings-btn" :disabled="backupBusy" @click="chooseBackupDir">
                  {{ ui.browse }}
                </button>
              </div>
            </div>
            <div class="settings-note full">{{ ui.backupHint }}</div>
            <div class="settings-field full">
              <div class="vault-row">
                <button type="button" class="settings-btn" :disabled="backupBusy || !backupDir" @click="backupNow">
                  {{ ui.backupNow }}
                </button>
                <label class="backup-keep">
                  {{ ui.backupSchedule }}
                  <select v-model.number="backupIntervalHours" @change="saveBackupSettings">
                    <option :value="0">{{ ui.backupScheduleManual }}</option>
                    <option :value="24">{{ ui.backupScheduleDaily }}</option>
                    <option :value="168">{{ ui.backupScheduleWeekly }}</option>
                    <option :value="720">{{ ui.backupScheduleMonthly }}</option>
                  </select>
                </label>
                <label class="backup-keep">
                  {{ ui.backupKeep }}
                  <input v-model.number="backupKeep" type="number" min="1" max="99" @change="saveBackupSettings" />
                </label>
                <span v-if="backupStatus" class="vault-status">{{ backupStatus }}</span>
              </div>
            </div>
            <ul v-if="backupEntries.length" class="backup-list">
              <li v-for="entry in backupEntries" :key="entry.path">
                <span class="backup-name">{{ entry.name }}</span>
                <span class="backup-size">{{ formatFileSize(entry.size) }}</span>
                <button type="button" class="settings-btn" :disabled="backupBusy" @click="restoreBackup(entry)">
                  {{ ui.backupRestore }}
                </button>
              </li>
            </ul>

            <!-- Knowledge API: expose the library to external MCP harnesses. -->
            <label class="settings-field full settings-toggle">
              <input v-model="knowledgeApi.enabled" type="checkbox" :disabled="knowledgeApiBusy" @change="saveKnowledgeApi" />
              <span>{{ ui.knowledgeApi }}</span>
            </label>
            <div class="settings-note full">{{ ui.knowledgeApiHint }}</div>
            <div class="settings-note full knowledge-api-warning">⚠️ {{ ui.knowledgeApiWarning }}</div>

            <template v-if="knowledgeApi.enabled">
              <div class="settings-field full">
                <div class="vault-row">
                  <label class="backup-keep">
                    {{ ui.knowledgeApiPort }}
                    <input v-model.number="knowledgeApi.port" type="number" min="1024" max="65535" :disabled="knowledgeApiBusy" @change="saveKnowledgeApi" />
                  </label>
                  <span :class="['api-status', knowledgeApi.running ? 'is-up' : 'is-down']">
                    {{ knowledgeApi.running ? `${ui.knowledgeApiRunning} · ${knowledgeApi.url}` : ui.knowledgeApiStopped }}
                  </span>
                </div>
              </div>
              <div v-if="knowledgeApi.error" class="settings-note full knowledge-api-error">{{ knowledgeApi.error }}</div>

              <div class="settings-field full">
                <span class="settings-label">{{ ui.knowledgeApiToken }}</span>
                <div class="vault-row">
                  <code class="api-token">{{ knowledgeApiTokenDisplay }}</code>
                  <button type="button" class="settings-btn" @click="knowledgeApiTokenVisible = !knowledgeApiTokenVisible">
                    {{ knowledgeApiTokenVisible ? ui.hide : ui.show }}
                  </button>
                  <button type="button" class="settings-btn" @click="copyText(knowledgeApi.token, ui.knowledgeApiCopied)">
                    {{ ui.copy }}
                  </button>
                  <button type="button" class="settings-btn" :disabled="knowledgeApiBusy" @click="rotateKnowledgeApiToken">
                    {{ ui.knowledgeApiRotate }}
                  </button>
                </div>
              </div>
              <div class="settings-note full">{{ ui.knowledgeApiRotateHint }}</div>

              <div class="settings-field full">
                <span class="settings-label">{{ ui.knowledgeApiConnect }}</span>
                <div class="vault-row">
                  <button type="button" class="settings-btn" @click="copyText(knowledgeApiClaudeCommand, ui.knowledgeApiCopied)">
                    {{ ui.knowledgeApiCopyClaude }}
                  </button>
                  <button type="button" class="settings-btn" @click="copyText(knowledgeApiCodexCommand, ui.knowledgeApiCopied)">
                    {{ ui.knowledgeApiCopyCodex }}
                  </button>
                </div>
              </div>
            </template>
          </div>

          <div v-if="settingsSection === 'chat'" class="settings-panel provider-settings-panel">
            <section class="local-agents-card">
              <div class="local-agents-head">
                <div>
                  <div class="settings-section-title compact">{{ ui.localAgents }}</div>
                  <div class="settings-note">{{ ui.localAgentsNote }}</div>
                </div>
                <button
                  type="button"
                  class="local-agents-recheck"
                  :disabled="localAgentChecking"
                  @click="refreshLocalAgentStatus"
                >{{ localAgentChecking ? ui.localAgentChecking : ui.localAgentRecheck }}</button>
              </div>
              <div class="local-agents-rows">
                <div
                  v-for="agent in localAgentStatus"
                  :key="agent.kind"
                  class="local-agent-row"
                >
                  <label class="local-agent-pref" :title="ui.localAgentPreferred">
                    <input
                      type="radio"
                      name="preferred-local-agent"
                      :value="agent.kind"
                      :checked="preferredLocalAgent === agent.kind"
                      @change="preferredLocalAgent = agent.kind"
                    />
                  </label>
                  <span class="local-agent-name">{{ agent.label }}</span>
                  <span
                    class="local-agent-pill"
                    :class="agent.installed ? 'ok' : 'missing'"
                  >{{ agent.installed ? (ui.localAgentInstalled + (agent.version ? ' · ' + agent.version : '')) : ui.localAgentNotInstalled }}</span>
                  <span class="local-agents-spacer"></span>
                  <span
                    v-if="localAgentTest[agent.kind] && !localAgentTest[agent.kind].testing"
                    class="local-agent-test-result"
                    :class="localAgentTest[agent.kind].ok ? 'ok' : 'missing'"
                    :title="localAgentTest[agent.kind].message"
                  >{{ localAgentTest[agent.kind].ok ? '✓' : '✗' }} {{ localAgentTest[agent.kind].message }}</span>
                  <button
                    v-if="agent.installed"
                    type="button"
                    class="local-agent-install"
                    :disabled="localAgentTest[agent.kind] && localAgentTest[agent.kind].testing"
                    @click="testLocalAgent(agent.kind)"
                  >{{ localAgentTest[agent.kind] && localAgentTest[agent.kind].testing ? ui.localAgentTesting : ui.localAgentTest }}</button>
                  <button
                    v-if="!agent.installed"
                    type="button"
                    class="local-agent-install"
                    @click="openExternalLink(agent.installUrl)"
                  >{{ ui.localAgentInstall }} ↗</button>
                </div>
                <div v-if="!localAgentStatus.length" class="settings-note">{{ ui.localAgentNone }}</div>
              </div>
            </section>
            <aside class="provider-list">
              <div class="provider-list-head">
                <div>
                  <div class="settings-section-title compact">{{ ui.chatProvidersNav }}</div>
                  <div class="settings-note">{{ ui.chatProvidersListNote }}</div>
                </div>
                <button type="button" class="provider-add-btn" @click="createNewProvider">+</button>
              </div>

              <div
                v-for="provider in editableProviders"
                :key="providerEditKey(provider)"
                class="provider-list-item"
                :class="{ active: selectedProviderEditKey === providerEditKey(provider) }"
              >
                <button
                  type="button"
                  class="provider-list-select"
                  @click="selectModelProvider(provider)"
                >
                  <span class="provider-list-name">{{ providerListName(provider) }}</span>
                  <span class="provider-list-meta">{{ providerListMeta(provider) }}</span>
                  <span v-if="provider.isDefault" class="provider-list-badge">{{ ui.defaultProviderShort }}</span>
                </button>
                <div class="provider-list-actions">
                  <button
                    v-if="!provider.isDefault"
                    type="button"
                    class="provider-list-default"
                    :disabled="settingsStatus === 'saving'"
                    :title="ui.setDefaultProvider"
                    :aria-label="ui.setDefaultProvider"
                    @click.stop="setDefaultModelProvider(provider)"
                  >
                    {{ ui.defaultShort }}
                  </button>
                  <button
                    type="button"
                    class="provider-list-delete"
                    :disabled="settingsStatus === 'saving'"
                    :title="ui.remove"
                    :aria-label="ui.remove"
                    @click.stop="removeEditableProvider(provider)"
                  >
                    ×
                  </button>
                </div>
              </div>
            </aside>

            <section class="provider-detail">
              <div class="provider-detail-head">
                <div>
                  <div class="settings-section-title compact">{{ ui.chatModelProviderSection }}</div>
                  <div class="settings-note">{{ ui.chatModelProviderNote }}</div>
                </div>
                <div class="provider-summary">{{ providerConnectionSummary }}</div>
              </div>

              <div class="provider-form-grid">
                <label class="settings-field">
                  <span>{{ ui.providerName }}</span>
                  <input v-model="providerForm.name" type="text" />
                </label>

                <label class="settings-field">
                  <span>{{ ui.providerType }}</span>
                  <select v-model="providerForm.providerType">
                    <option value="openai-compatible">OpenAI-compatible</option>
                    <option value="openai">OpenAI</option>
                    <option value="deepseek">DeepSeek</option>
                    <option value="openrouter">OpenRouter</option>
                  </select>
                </label>

                <label class="settings-field full">
                  <span>{{ ui.baseUrl }}</span>
                  <input v-model="providerForm.baseUrl" type="url" :placeholder="providerTypePreset.baseUrl" />
                </label>

                <label class="settings-field">
                  <span>{{ ui.apiKey }}</span>
                  <input v-model="providerForm.apiKey" type="password" :placeholder="ui.apiKeyPlaceholder" />
                </label>

                <div v-if="providerForm.hasApiKey" class="settings-note">{{ ui.apiKeySaved }}</div>
              </div>

              <div class="models-head">
                <div>
                  <div class="settings-section-title compact">{{ ui.availableModels }}</div>
                  <div class="settings-note">{{ ui.modelsForProviderNote }}</div>
                </div>
                <div class="models-head-actions">
                  <button
                    type="button"
                    class="settings-btn primary-subtle"
                    :disabled="modelFetchStatus === 'fetching' || settingsStatus === 'saving' || !providerForm.baseUrl"
                    @click="fetchProviderModels"
                  >
                    {{ modelFetchStatus === 'fetching' ? `${ui.fetchModels}...` : ui.fetchModels }}
                  </button>
                  <button
                    type="button"
                    class="settings-btn"
                    :disabled="settingsStatus === 'saving'"
                    @click="addProviderModel"
                  >
                    {{ ui.addManually }}
                  </button>
                </div>
              </div>

              <div class="model-table">
                <div class="model-table-head" aria-hidden="true">
                  <span>{{ ui.defaultShort }}</span>
                  <span>{{ ui.modelNickname }}</span>
                  <span>{{ ui.modelId }}</span>
                  <span>{{ ui.contextWindow }}</span>
                  <span>{{ ui.modelCapabilities }}</span>
                  <span>{{ ui.enabled }}</span>
                  <span></span>
                </div>

                <article
                  v-for="(model, index) in providerForm.models"
                  :key="model.key || `draft-${index}`"
                  class="model-row"
                >
                  <label class="model-default-cell" :title="ui.defaultChatModel">
                    <input
                      :checked="providerForm.defaultModelKey === model.key"
                      type="radio"
                      name="default-chat-model"
                      @change="setDefaultProviderModel(model)"
                    />
                    <span>{{ index + 1 }}</span>
                  </label>

                  <label class="model-field-cell">
                    <span>{{ ui.modelNickname }}</span>
                    <input v-model="model.nickname" type="text" :placeholder="model.modelId || ui.modelNickname" />
                  </label>

                  <label class="model-field-cell model-id-cell">
                    <span>{{ ui.modelId }}</span>
                    <input v-model="model.modelId" type="text" :placeholder="providerTypePreset.model || 'gpt-4o / deepseek-chat'" />
                  </label>

                  <label class="model-field-cell model-context-cell" :title="ui.contextWindowHint">
                    <span>{{ ui.contextWindow }}</span>
                    <input
                      v-model.number="model.contextWindowOverride"
                      type="number"
                      min="1024"
                      step="1024"
                      :placeholder="contextWindowPlaceholder(model)"
                    />
                  </label>

                  <div class="model-capability-cell" :aria-label="ui.modelCapabilities">
                    <span class="capability-chip fixed">{{ ui.capabilityText }}</span>
                    <button
                      v-for="capability in MODEL_CAPABILITY_OPTIONS"
                      :key="`${model.key || index}-${capability}`"
                      type="button"
                      class="capability-chip"
                      :class="{ active: model.capabilities.includes(capability) }"
                      @click="toggleModelCapability(model, capability)"
                    >
                      {{ modelCapabilityLabel(capability) }}
                    </button>
                  </div>

                  <label class="model-enabled-cell" :title="ui.enabled">
                    <input v-model="model.enabled" type="checkbox" />
                    <span>{{ ui.enabled }}</span>
                  </label>

                  <button
                    type="button"
                    class="model-remove-btn"
                    :disabled="providerForm.models.length === 1"
                    :title="ui.remove"
                    :aria-label="ui.remove"
                    @click="removeProviderModel(index)"
                  >
                    ×
                  </button>
                </article>
              </div>

              <div class="settings-inline-actions">
                <button
                  type="button"
                  class="settings-btn"
                  :disabled="providerTestStatus === 'testing' || settingsStatus === 'saving'"
                  @click="testChatModelProvider"
                >
                  {{ ui.testChatModel }}
                </button>
              </div>
            </section>
          </div>

          <div v-if="settingsSection === 'websearch'" class="settings-panel settings-body">
            <div class="settings-section-title full">{{ ui.webSearchSection }}</div>
            <div class="settings-note full">{{ ui.webSearchSectionNote }}</div>
            <label class="settings-field full">
              <span>{{ ui.exaApiKey }}</span>
              <input
                v-model="webSearchForm.apiKey"
                type="password"
                autocomplete="off"
                :placeholder="webSearchForm.hasApiKey ? ui.apiKeyPlaceholder : ui.exaApiKeyPlaceholder"
              />
            </label>
            <div v-if="webSearchForm.hasApiKey" class="settings-note full">{{ ui.exaApiKeySaved }}</div>
            <div v-else class="settings-note full">{{ ui.exaFallbackNote }}</div>
            <div class="settings-inline-actions">
              <button
                type="button"
                class="settings-btn"
                :disabled="webSearchForm.status === 'saving'"
                @click="saveWebSearchSettings"
              >
                {{ webSearchForm.status === 'saving' ? ui.saving : ui.saveWebSearchKey }}
              </button>
            </div>

            <div class="settings-section-title full">{{ ui.proxySection }}</div>
            <div class="settings-note full">{{ ui.proxySectionNote }}</div>
            <label class="settings-field full">
              <span>{{ ui.proxyUrl }}</span>
              <input
                v-model="proxyForm.proxyUrl"
                type="text"
                autocomplete="off"
                spellcheck="false"
                :placeholder="ui.proxyUrlPlaceholder"
              />
            </label>
            <div class="settings-inline-actions">
              <button
                type="button"
                class="settings-btn"
                :disabled="proxyForm.status === 'saving'"
                @click="saveProxySettings"
              >
                {{ proxyForm.status === 'saving' ? ui.saving : ui.saveProxy }}
              </button>
              <span v-if="proxyForm.status === 'saved'" class="settings-note">{{ ui.proxySaved }}</span>
            </div>
          </div>

          <div v-if="settingsSection === 'translation'" class="settings-panel settings-body">
            <div class="settings-section-title full">{{ ui.translationProviderSection }}</div>
            <label class="settings-field full">
              <span>{{ ui.translationProviderMode }}</span>
              <select v-model="translationProvider">
                <option value="google-web">{{ ui.translationProviderGoogleWeb }}</option>
                <option value="microsoft">{{ ui.translationProviderMicrosoft }}</option>
                <option value="llm">{{ ui.translationProviderLlm }}</option>
                <option value="local-placeholder">{{ ui.translationProviderPlaceholder }}</option>
              </select>
            </label>

            <div class="settings-note full">
              {{ translationProviderNote }}
            </div>

            <label class="settings-check full">
              <input v-model="translationFallbackEnabled" type="checkbox" />
              <span>{{ ui.translationFallback }}</span>
            </label>

            <div class="settings-note full">
              {{ ui.translationFallbackNote }}
            </div>

            <template v-if="translationProvider === 'microsoft'">
              <label class="settings-field full">
                <span>{{ ui.microsoftEndpoint }}</span>
                <input
                  v-model="microsoftForm.endpoint"
                  type="url"
                  placeholder="https://api.cognitive.microsofttranslator.com"
                />
              </label>

              <label class="settings-field">
                <span>{{ ui.microsoftRegion }}</span>
                <input v-model="microsoftForm.region" type="text" placeholder="eastus" />
              </label>

              <label class="settings-field">
                <span>{{ ui.apiKey }}</span>
                <input v-model="microsoftForm.apiKey" type="password" :placeholder="ui.apiKeyPlaceholder" />
              </label>

              <div v-if="microsoftForm.hasApiKey" class="settings-note full">{{ ui.apiKeySaved }}</div>
            </template>
          </div>
        </div>

        <div
          v-if="settingsStatus === 'saved' || settingsStatus === 'failed' || providerTestStatus !== 'idle' || modelFetchStatus !== 'idle'"
          class="settings-message"
          :class="{ failed: settingsStatus === 'failed' || providerTestStatus === 'failed' || modelFetchStatus === 'failed' }"
        >
          <template v-if="settingsStatus === 'failed'">
            {{ ui.settingsSaveFailed }}: {{ settingsError }}
          </template>
          <template v-else-if="settingsStatus === 'saved'">
            {{ ui.settingsSaved }}
          </template>
          <template v-else-if="providerTestStatus === 'testing'">
            {{ ui.testConnection }}...
          </template>
          <template v-else-if="providerTestStatus === 'succeeded'">
            {{ ui.providerTestSucceeded }}: {{ providerTestMessage }}
          </template>
          <template v-else-if="providerTestStatus === 'failed'">
            {{ ui.providerTestFailed }}: {{ providerTestMessage }}
          </template>
          <template v-else-if="modelFetchStatus === 'fetching'">
            {{ ui.fetchingModels }}
          </template>
          <template v-else-if="modelFetchStatus === 'succeeded'">
            {{ modelFetchMessage }}
          </template>
          <template v-else-if="modelFetchStatus === 'failed'">
            {{ ui.fetchModelsFailed }}: {{ modelFetchMessage }}
          </template>
        </div>

        <div class="settings-actions">
          <button type="button" class="settings-btn" @click="closeSettings">{{ ui.close }}</button>
          <button
            type="button"
            class="settings-btn"
            :disabled="providerTestStatus === 'testing' || settingsStatus === 'saving'"
            @click="testProviderSettings"
          >
            {{ ui.testConnection }}
          </button>
          <button
            type="button"
            class="settings-btn primary"
            :disabled="settingsStatus === 'saving'"
            @click="saveProviderSettings"
          >
            {{ settingsStatus === 'saving' ? `${ui.save}...` : ui.save }}
          </button>
        </div>
      </section>
    </div>
  </div>
</template>

<style scoped>
.app-shell {
  height: 100vh;
  background: var(--surface-0);
  display: flex;
  overflow: hidden;
  /* Positioning context for the floating Notes drawer. */
  position: relative;
}

/* Center column: optional knowledge card pinned above the reader. */
.reader-column {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
  min-height: 0;
}

/* Update notification: a small floating toast at the bottom-center. Notify-only
   — links out to the tags page; the user updates manually. */
.update-banner {
  position: absolute;
  z-index: 60;
  left: 50%;
  bottom: 20px;
  transform: translateX(-50%);
  display: flex;
  align-items: center;
  gap: 10px;
  max-width: calc(100vw - 48px);
  padding: 8px 10px 8px 14px;
  border: 1px solid var(--accent-line);
  border-radius: var(--r-pill);
  background: var(--surface-1);
  box-shadow: var(--shadow-overlay);
  backdrop-filter: blur(8px);
  font-size: 13px;
  color: var(--ink);
}

.update-banner-icon {
  color: var(--accent);
  font-weight: var(--w-medium);
}

.update-banner-text {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.update-banner-view {
  flex-shrink: 0;
  padding: 4px 12px;
  border: 1px solid var(--accent-line);
  border-radius: var(--r-pill);
  background: var(--accent-tint);
  color: var(--ink);
  font-size: 12px;
  font-weight: var(--w-strong);
  cursor: pointer;
}

.update-banner-view:hover {
  background: var(--accent-tint);
}

.update-banner-dismiss {
  flex-shrink: 0;
  width: 24px;
  height: 24px;
  display: grid;
  place-items: center;
  border: none;
  border-radius: var(--r-pill);
  background: transparent;
  color: var(--ink-3);
  font-size: 16px;
  line-height: 1;
  cursor: pointer;
}

.update-banner-dismiss:hover {
  background: var(--surface-hover-strong);
  color: var(--ink);
}

/* Notes drawer: an absolutely-positioned wrapper that floats over the Agent
   pane (does NOT squeeze the Reader). Styling the wrapper div here — rather than
   the NotesPane root — avoids a scoped-CSS specificity tie with NotesPane's own
   `.notes-shell { position: relative }`, which previously left the pane in the
   flex flow and pushed the Reader narrower. The NotesPane inside fills it. */
.notes-drawer {
  position: absolute;
  top: 0;
  right: 0;
  bottom: 0;
  width: 330px;
  z-index: 40;
  box-shadow: var(--shadow-overlay);
}

.notes-drawer > :deep(.notes-shell) {
  width: 100% !important;
  height: 100%;
}

.notes-drawer-enter-active,
.notes-drawer-leave-active {
  transition: transform var(--dur-slow) var(--ease), opacity var(--dur-slow) var(--ease);
}

.notes-drawer-enter-from,
.notes-drawer-leave-to {
  transform: translateX(100%);
  opacity: 0.4;
}

.async-panel-loading {
  min-width: 0;
  min-height: 0;
  flex: 1;
  position: relative;
  overflow: hidden;
  background: var(--surface-1);
}

.async-panel-loading::before {
  content: "";
  position: absolute;
  inset: 18px;
  border-radius: var(--r-sm);
  border: 1px solid var(--line);
  background:
    linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.055), transparent),
    rgba(255, 255, 255, 0.025);
  background-size: 220% 100%;
  animation: async-panel-shimmer 1.25s ease-in-out infinite;
}

@keyframes async-panel-shimmer {
  0% {
    background-position: 180% 0;
  }

  100% {
    background-position: -80% 0;
  }
}

.left-collapse-btn {
  width: 24px;
  height: 34px;
  margin-left: -12px;
  margin-right: -12px;
  align-self: flex-start;
  margin-top: 5px;
  border-radius: var(--r-pill);
  border: 1px solid var(--line);
  background: var(--surface-2);
  color: var(--ink-2);
  cursor: pointer;
  z-index: 8;
  display: grid;
  place-items: center;
  padding: 0;
  line-height: 1;
}

.left-collapse-btn:hover {
  color: var(--ink);
  background: var(--surface-hover-strong);
}

.drag-handle {
  width: 10px;
  margin-left: -5px;
  cursor: col-resize;
  position: relative;
  z-index: 3;
}

.drag-handle::before {
  content: '';
  position: absolute;
  left: 50%;
  top: 14px;
  bottom: 14px;
  width: 2px;
  border-radius: var(--r-pill);
  background: var(--surface-hover-strong);
  transform: translateX(-50%);
}

.settings-backdrop,
.confirm-backdrop {
  position: fixed;
  inset: 0;
  z-index: 20;
  display: grid;
  place-items: center;
  padding: 24px;
  background: var(--scrim);
}

.confirm-backdrop {
  z-index: 22;
}

.confirm-modal {
  width: min(430px, 100%);
  overflow: hidden;
  border: 1px solid var(--line);
  border-radius: var(--r-lg);
  background: var(--surface-2);
  box-shadow: var(--shadow-overlay);
}

.confirm-head,
.confirm-actions {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 16px 18px;
}

.confirm-head {
  border-bottom: 1px solid var(--line);
}

.confirm-title {
  color: var(--ink);
  font-size: 15px;
  font-weight: var(--w-strong);
}

.confirm-close {
  width: 30px;
  height: 30px;
  border-radius: var(--r-pill);
  border: 1px solid var(--line);
  background: var(--surface-wash);
  color: var(--ink-2);
  cursor: pointer;
  font-size: 18px;
}

.confirm-close:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.confirm-body {
  padding: 18px;
  color: var(--ink-2);
  font-size: 13px;
  line-height: 1.55;
}

.confirm-body p {
  margin: 0;
}

.confirm-target {
  margin-top: 14px;
  padding: 10px 12px;
  border-radius: var(--r-md);
  border: 1px solid var(--line);
  background: var(--surface-wash);
  color: var(--ink);
  font-size: 12px;
  font-weight: var(--w-medium);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.confirm-error {
  margin-top: 12px;
  padding: 10px 12px;
  border-radius: var(--r-md);
  border: 1px solid var(--danger-line);
  background: var(--danger-tint);
  color: var(--danger);
  font-size: 12px;
}

.confirm-actions {
  justify-content: flex-end;
  border-top: 1px solid var(--line);
}

.confirm-btn {
  min-width: 86px;
  border: 1px solid var(--line);
  border-radius: var(--r-pill);
  background: var(--surface-wash);
  color: var(--ink);
  cursor: pointer;
  padding: 8px 13px;
  font-weight: var(--w-medium);
}

.confirm-btn:hover:not(:disabled) {
  background: var(--surface-hover-strong);
}

.confirm-btn.danger {
  border-color: var(--danger-line);
  background: var(--danger-tint);
  color: var(--danger);
}

.confirm-btn.danger:hover:not(:disabled) {
  background: var(--danger-tint);
}

.confirm-btn:disabled {
  opacity: 0.5;
  cursor: wait;
}

.settings-modal {
  width: min(1080px, 100%);
  max-height: min(720px, calc(100vh - 48px));
  display: flex;
  flex-direction: column;
  overflow: hidden;
  border: 1px solid var(--line);
  border-radius: var(--r-xl);
  background: var(--surface-2);
  box-shadow: var(--shadow-overlay);
}

.settings-head,
.settings-actions {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  padding: 18px 20px;
  flex-shrink: 0;
}

.settings-head {
  border-bottom: 1px solid var(--line);
}

.settings-title {
  font-size: 17px;
  font-weight: var(--w-strong);
  color: var(--ink);
}

.settings-subtitle {
  margin-top: 4px;
  font-size: 12px;
  color: var(--ink-3);
}

.settings-close {
  width: 32px;
  height: 32px;
  border-radius: var(--r-pill);
  border: 1px solid var(--line);
  background: var(--surface-wash);
  color: var(--ink-2);
  cursor: pointer;
  font-size: 20px;
}

/* The two columns scroll independently. Scrolling the whole layout — which is
   what this used to do — dragged the section list off the top of the dialog
   along with the content, and gave the dialog a single scrollbar running its
   full height right against the rounded border. */
.settings-layout {
  display: grid;
  grid-template-columns: 190px minmax(0, 1fr);
  flex: 1 1 auto;
  min-height: 0;
  overflow: hidden;
  /* A gutter for the panel's scrollbar. Without it the bar sits flush against
     the dialog's rounded frame and reads as being drawn on top of the dialog
     rather than inside its content. */
  padding-right: 6px;
}

.settings-nav {
  padding: 16px 12px;
  border-right: 1px solid var(--line);
  background: var(--surface-wash);
  display: flex;
  flex-direction: column;
  gap: 8px;
  /* min-height on a grid item, or the track refuses to shrink and scroll. */
  min-height: 0;
  overflow-y: auto;
}

.settings-nav-item {
  width: 100%;
  border: 1px solid transparent;
  border-radius: var(--r-md);
  background: transparent;
  color: var(--ink-2);
  cursor: pointer;
  padding: 10px 12px;
  text-align: left;
}

.settings-nav-item span,
.settings-nav-item small {
  display: block;
}

.settings-nav-item span {
  color: var(--ink);
  font-size: 13px;
  font-weight: var(--w-medium);
}

.settings-nav-item small {
  margin-top: 4px;
  color: var(--ink-3);
  font-size: 11px;
  line-height: 1.35;
}

.settings-nav-item.active {
  border-color: var(--accent-line);
  background: var(--accent-tint);
}

/* Every section (general / chat / websearch / translation) carries this class,
   so putting the scroll here covers all four. */
.settings-panel {
  min-width: 0;
  min-height: 0;
  overflow-y: auto;
}

.settings-body {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 14px;
  /* Right padding is trimmed by the layout's 6px scrollbar gutter, so the
     content still sits optically centred between the nav and the frame. */
  padding: 20px 14px 20px 20px;
}

.provider-settings-panel {
  display: grid;
  grid-template-columns: 230px minmax(0, 1fr);
  min-width: 0;
}

/* Local-agent (Codex / Claude) status card, spanning both grid columns above the
   provider list + form. */
.local-agents-card {
  grid-column: 1 / -1;
  padding: 14px 16px;
  border-bottom: 1px solid var(--line);
  background: var(--surface-wash);
}

.local-agents-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 10px;
}

.local-agents-recheck {
  flex-shrink: 0;
  border: 1px solid var(--line);
  border-radius: var(--r-sm);
  background: transparent;
  color: var(--ink-2);
  font-size: 12px;
  padding: 4px 10px;
  cursor: pointer;
}

.local-agents-recheck:hover:not(:disabled) {
  color: var(--ink);
  border-color: var(--accent-line);
}

.local-agents-recheck:disabled {
  opacity: 0.5;
  cursor: progress;
}

.local-agents-rows {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.local-agent-row {
  display: flex;
  align-items: center;
  gap: 10px;
}

.local-agent-pref {
  display: inline-flex;
  align-items: center;
  cursor: pointer;
}

.local-agent-name {
  font-size: 13px;
  color: var(--ink);
  min-width: 96px;
}

.local-agent-pill {
  font-size: 11px;
  padding: 2px 9px;
  border-radius: var(--r-pill);
}

.local-agent-pill.ok {
  color: var(--success);
  background: var(--success-tint);
}

.local-agent-pill.missing {
  color: var(--ink-3);
  background: var(--surface-hover);
}

.local-agents-spacer {
  flex: 1;
}

.local-agent-test-result {
  font-size: 11px;
  max-width: 240px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.local-agent-test-result.ok {
  color: var(--success);
}

.local-agent-test-result.missing {
  color: var(--danger);
}

.local-agent-install {
  border: none;
  background: transparent;
  color: var(--accent);
  font-size: 12px;
  cursor: pointer;
  padding: 0;
}

.local-agent-install:disabled {
  color: var(--ink-3);
  cursor: default;
}

.provider-list {
  padding: 18px 14px;
  border-right: 1px solid var(--line);
  display: flex;
  flex-direction: column;
  gap: 10px;
  min-width: 0;
}

.provider-list-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 10px;
  margin-bottom: 4px;
}

.settings-section-title.compact {
  padding-top: 0;
  margin-top: 0;
  border-top: none;
}

.provider-add-btn {
  width: 30px;
  height: 30px;
  flex: 0 0 auto;
  border-radius: var(--r-pill);
  border: 1px solid var(--line);
  background: var(--surface-wash);
  color: var(--ink);
  cursor: pointer;
  font-size: 18px;
  line-height: 1;
}

.provider-list-item {
  position: relative;
  width: 100%;
  min-height: 72px;
  border-radius: var(--r-lg);
  border: 1px solid var(--line);
  background: var(--surface-wash);
  color: var(--ink-2);
  padding: 0;
  overflow: hidden;
}

.provider-list-item.active {
  border-color: var(--accent-line);
  background: var(--accent-tint);
}

.provider-list-select {
  width: 100%;
  min-height: 70px;
  border: none;
  background: transparent;
  color: inherit;
  cursor: pointer;
  padding: 12px 74px 12px 12px;
  text-align: left;
}

.provider-list-actions {
  position: absolute;
  top: 9px;
  right: 8px;
  display: flex;
  align-items: center;
  gap: 5px;
}

.provider-list-default,
.provider-list-delete {
  border: 1px solid transparent;
  background: transparent;
  color: var(--ink-3);
  cursor: pointer;
}

.provider-list-default {
  min-height: 26px;
  padding: 0 7px;
  border-radius: var(--r-pill);
  font-size: 10px;
  font-weight: var(--w-medium);
}

.provider-list-default:hover:not(:disabled) {
  border-color: var(--accent-line);
  background: var(--accent-tint);
  color: var(--accent);
}

.provider-list-delete {
  width: 26px;
  height: 26px;
  border-radius: var(--r-pill);
  font-size: 18px;
  line-height: 1;
}

.provider-list-delete:hover:not(:disabled) {
  border-color: var(--danger-line);
  background: var(--danger-tint);
  color: var(--danger);
}

.provider-list-default:disabled,
.provider-list-delete:disabled {
  cursor: not-allowed;
  opacity: 0.3;
}

.provider-list-name,
.provider-list-meta {
  display: block;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.provider-list-name {
  color: var(--ink);
  font-size: 13px;
  font-weight: var(--w-medium);
}

.provider-list-meta {
  margin-top: 6px;
  color: var(--ink-3);
  font-size: 11px;
}

.provider-list-badge {
  display: inline-flex;
  margin-top: 8px;
  padding: 3px 7px;
  border-radius: var(--r-pill);
  background: var(--accent-tint);
  color: var(--accent);
  font-size: 10px;
  font-weight: var(--w-medium);
}

.provider-detail {
  min-width: 0;
  padding: 20px;
}

.provider-detail-head,
.models-head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 16px;
  margin-bottom: 16px;
}

.models-head-actions {
  display: inline-flex;
  flex-wrap: wrap;
  justify-content: flex-end;
  gap: 8px;
}

.provider-summary {
  max-width: 360px;
  min-height: 30px;
  border-radius: var(--r-pill);
  border: 1px solid var(--line);
  background: var(--surface-wash);
  color: var(--ink-3);
  padding: 7px 10px;
  font-size: 11px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.provider-form-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 14px;
  padding-bottom: 18px;
  margin-bottom: 18px;
  border-bottom: 1px solid var(--line);
}

.settings-field,
.settings-check {
  display: flex;
  flex-direction: column;
  gap: 7px;
  min-width: 0;
}

.settings-field.full,
.settings-check.full,
.settings-note.full {
  grid-column: 1 / -1;
}

/* .settings-toggle was used in the template but never actually styled, so it
   inherited .settings-field's column flex: the checkbox became a full-width
   flex item, 840px wide, floating above its own label. It was always wrong —
   just hard to see as a dark box on a dark panel. Under color-scheme: light the
   platform paints a filled blue square and it is unmissable. */
.settings-field.settings-toggle {
  flex-direction: row;
  align-items: center;
  gap: var(--gap-3);
  cursor: pointer;
}

.settings-field.settings-toggle input[type='checkbox'] {
  flex: none;
  width: 15px;
  height: 15px;
  margin: 0;
  accent-color: var(--accent);
}

/* Segmented control. The track carries the hairline so the three buttons read
   as one control rather than three loose chips. */
.theme-choice {
  display: inline-flex;
  align-self: start;
  padding: 2px;
  gap: 2px;
  border-radius: var(--r-md);
  background: var(--surface-field);
  box-shadow: var(--shadow-hairline);
}

.theme-choice-btn {
  display: inline-flex;
  align-items: center;
  gap: var(--gap-2);
  min-height: 28px;
  padding: 0 var(--gap-4);
  border: none;
  border-radius: var(--r-sm);
  background: transparent;
  color: var(--ink-2);
  font-size: var(--fs-body);
  cursor: pointer;
  transition:
    background var(--dur-fast) var(--ease),
    color var(--dur-fast) var(--ease);
}

.theme-choice-btn svg {
  width: 14px;
  height: 14px;
  flex: none;
}

.theme-choice-btn:hover {
  color: var(--ink);
}

/* The selected segment is a raised tile on the recessed track — the same
   figure/ground trick the platform controls use, so it reads as pressed in
   both themes without needing an accent fill. */
.theme-choice-btn.active {
  background: var(--surface-2);
  color: var(--ink);
  box-shadow: var(--shadow-btn);
}

.theme-choice-btn:focus-visible {
  outline: none;
  box-shadow: var(--ring-focus);
}

.settings-section-title.full,
.settings-inline-actions.full,
.model-table.full {
  grid-column: 1 / -1;
}

.settings-section-title {
  padding-top: 6px;
  margin-top: 4px;
  border-top: 1px solid var(--line);
  font-size: 12px;
  color: var(--ink-2);
  text-transform: uppercase;
  letter-spacing: var(--tracking-caps);
}

.settings-field span,
.settings-check span {
  font-size: 12px;
  color: var(--ink-2);
}

.settings-field input,
.settings-field select {
  width: 100%;
  min-height: 40px;
  border-radius: var(--r-md);
  border: 1px solid var(--line);
  background: var(--surface-hover);
  color: var(--ink);
  padding: 0 12px;
  outline: none;
}

.settings-check {
  flex-direction: row;
  align-items: center;
  min-height: 32px;
}

.settings-check.inline {
  min-height: 28px;
}

.settings-check input {
  width: 16px;
  height: 16px;
  accent-color: var(--accent);
}

.settings-note,
.settings-message {
  color: var(--ink-3);
  font-size: 12px;
  line-height: 1.45;
}

.settings-inline-actions {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  justify-content: flex-end;
}

.model-table {
  display: flex;
  flex-direction: column;
  overflow-x: auto;
  overflow-y: hidden;
  border: 1px solid var(--line);
  border-radius: var(--r-lg);
  background: var(--surface-wash);
}

.model-table-head,
.model-row {
  display: grid;
  grid-template-columns: 48px minmax(140px, 1fr) minmax(190px, 1.1fr) 116px minmax(260px, 1.25fr) 48px 34px;
  gap: 8px;
  align-items: center;
  min-width: 880px;
}

.model-table-head {
  min-height: 36px;
  padding: 0 10px;
  border-bottom: 1px solid var(--line);
  color: var(--ink-3);
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0;
}

.model-row {
  min-height: 68px;
  padding: 10px;
}

.model-row + .model-row {
  border-top: 1px solid var(--line);
}

.model-default-cell,
.model-enabled-cell {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  color: var(--ink-2);
  font-size: 12px;
}

.model-default-cell {
  justify-content: center;
}

.model-default-cell input,
.model-enabled-cell input {
  width: 16px;
  height: 16px;
  accent-color: var(--accent);
}

.model-default-cell span {
  color: var(--ink-3);
  font-size: 11px;
}

.model-enabled-cell span {
  display: none;
}

.model-field-cell {
  display: flex;
  flex-direction: column;
  gap: 6px;
  min-width: 0;
}

.model-field-cell span {
  display: none;
  color: var(--ink-3);
  font-size: 11px;
}

.model-field-cell input {
  width: 100%;
  min-width: 0;
  min-height: 36px;
  border-radius: var(--r-md);
  border: 1px solid transparent;
  background: var(--surface-hover);
  color: var(--ink);
  padding: 0 11px;
  outline: none;
}

.model-field-cell input:focus {
  border-color: var(--accent-line);
  background: var(--surface-hover-strong);
}

.model-capability-cell {
  display: flex;
  flex-wrap: wrap;
  gap: 5px;
  min-width: 0;
}

.capability-chip {
  min-height: 26px;
  border-radius: var(--r-pill);
  border: 1px solid var(--line);
  background: var(--surface-wash);
  color: var(--ink-2);
  cursor: pointer;
  padding: 0 8px;
  font-size: 11px;
  line-height: 1;
  white-space: nowrap;
}

.capability-chip.fixed {
  display: inline-flex;
  align-items: center;
  cursor: default;
  color: var(--ink-3);
}

.capability-chip.active {
  border-color: var(--accent-line);
  background: var(--accent-tint);
  color: var(--ink);
}

.model-remove-btn {
  width: 30px;
  height: 30px;
  border-radius: var(--r-pill);
  border: 1px solid transparent;
  background: transparent;
  color: var(--ink-3);
  cursor: pointer;
  font-size: 20px;
  line-height: 1;
}

.model-remove-btn:hover:not(:disabled) {
  border-color: var(--danger-line);
  background: var(--danger-tint);
  color: var(--danger);
}

.model-remove-btn:disabled {
  cursor: not-allowed;
  opacity: 0.35;
}

.settings-message {
  margin: 12px 20px 0;
  padding: 10px 12px;
  border-radius: var(--r-md);
  border: 1px solid var(--accent-line);
  background: var(--accent-tint);
  color: var(--ink-2);
  flex-shrink: 0;
}

.settings-message.failed {
  border-color: var(--danger-line);
  background: var(--danger-tint);
  color: var(--danger);
}

.settings-actions {
  justify-content: flex-end;
  border-top: 1px solid var(--line);
}

.settings-btn {
  min-height: 38px;
  border-radius: var(--r-md);
  border: 1px solid var(--line);
  background: var(--surface-wash);
  color: var(--ink-2);
  cursor: pointer;
  padding: 0 14px;
}

/* Notes vault: path field beside its buttons, wrapping on a narrow panel. */
.vault-row {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
  width: 100%;
  min-width: 0;
}

.vault-input {
  flex: 1 1 240px;
  min-width: 0;
  min-height: 38px;
  padding: 0 12px;
  border-radius: var(--r-md);
  border: 1px solid var(--line);
  background: var(--surface-wash);
  color: var(--ink);
  outline: none;
  font-size: 13px;
}

.vault-input:focus {
  border-color: var(--accent-line);
}

.vault-status {
  font-size: 12px;
  color: var(--ink-3);
}

.backup-keep {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-size: 12px;
  color: var(--ink-3);
}

.backup-keep select {
  min-height: 32px;
  padding: 0 8px;
  border-radius: var(--r-md);
  border: 1px solid var(--line);
  background: var(--surface-wash);
  color: var(--ink);
  outline: none;
}

.backup-keep input {
  width: 56px;
  min-height: 32px;
  padding: 0 8px;
  border-radius: var(--r-md);
  border: 1px solid var(--line);
  background: var(--surface-wash);
  color: var(--ink);
  outline: none;
}

.backup-list {
  grid-column: 1 / -1;
  list-style: none;
  margin: 4px 0 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.backup-list li {
  display: flex;
  align-items: center;
  gap: 10px;
  min-width: 0;
}

.backup-name {
  flex: 1;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-size: 13px;
  color: var(--ink);
  font-variant-numeric: tabular-nums;
}

.backup-size {
  flex: 0 0 auto;
  font-size: 12px;
  color: var(--ink-3);
}

/* Knowledge API: the warning has to read as a warning, since enabling this
   widens the app's trust boundary to every process on the machine. */
.knowledge-api-warning {
  color: var(--warning);
}

.knowledge-api-error {
  color: var(--danger);
}

.api-status {
  font-size: 12px;
  font-variant-numeric: tabular-nums;
}

.api-status.is-up {
  color: var(--success);
}

.api-status.is-down {
  color: var(--ink-3);
}

.api-token {
  flex: 1 1 220px;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  padding: 8px 12px;
  border-radius: var(--r-md);
  border: 1px solid var(--line);
  background: var(--surface-wash);
  color: var(--ink);
  font-family: var(--font-mono);
  font-size: 12px;
}

.settings-btn.primary {
  background: var(--accent-tint);
  border-color: var(--accent-line);
  color: var(--ink);
}

.settings-btn.primary-subtle {
  background: var(--accent-tint);
  border-color: var(--accent-line);
  color: var(--ink);
}

.settings-btn:disabled {
  opacity: 0.55;
  cursor: not-allowed;
}

@media (max-width: 720px) {
  .settings-layout,
  .provider-settings-panel {
    grid-template-columns: 1fr;
  }

  .settings-nav {
    border-right: none;
    border-bottom: 1px solid var(--line);
    flex-direction: row;
    overflow-x: auto;
  }

  .settings-nav-item {
    min-width: 160px;
  }

  .provider-list {
    border-right: none;
    border-bottom: 1px solid var(--line);
  }

  .provider-detail-head,
  .models-head {
    flex-direction: column;
  }

  .provider-summary {
    max-width: 100%;
    width: 100%;
  }

  .provider-form-grid {
    grid-template-columns: 1fr;
  }

  .settings-body {
    grid-template-columns: 1fr;
  }

  .model-table {
    border-radius: var(--r-lg);
  }

  .model-table-head {
    display: none;
  }

  .model-row {
    grid-template-columns: 34px 1fr 30px;
    gap: 10px;
    align-items: start;
    min-width: 0;
  }

  .model-field-cell,
  .model-capability-cell,
  .model-enabled-cell {
    grid-column: 2 / 3;
  }

  .model-default-cell {
    grid-column: 1 / 2;
    grid-row: 1 / 4;
    padding-top: 26px;
  }

  .model-remove-btn {
    grid-column: 3 / 4;
    grid-row: 1;
  }

  .model-field-cell span {
    display: block;
  }

  .model-enabled-cell span {
    display: inline;
  }
}
</style>
