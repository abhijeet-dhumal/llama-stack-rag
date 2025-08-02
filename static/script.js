// Global state
let currentTab = 'upload';
let uploadQueue = [];
let isProcessing = false;

// DOM elements
const elements = {
    // Navigation
    navTabs: document.querySelectorAll('.nav-tab'),
    sections: document.querySelectorAll('.section'),
    
    // Status
    statusDot: document.getElementById('status-dot'),
    statusText: document.getElementById('status-text'),
    
    // Upload
    uploadBox: document.getElementById('upload-box'),
    fileInput: document.getElementById('file-input'),
    uploadQueue: document.getElementById('upload-queue'),
    queueList: document.getElementById('queue-list'),
    
    // Query
    queryInput: document.getElementById('query-input'),
    queryBtn: document.getElementById('query-btn'),
    contextLimit: document.getElementById('context-limit'),
    queryResults: document.getElementById('query-results'),
    
    // Dashboard
    docCount: document.getElementById('doc-count'),
    llmModel: document.getElementById('llm-model'),
    embeddingModel: document.getElementById('embedding-model'),
    pipelineStatus: document.getElementById('pipeline-status'),
    dbStatus: document.getElementById('db-status'),
    refreshStatsBtn: document.getElementById('refresh-stats'),
    clearDocsBtn: document.getElementById('clear-docs'),
    documentsContainer: document.getElementById('documents-container'),
    
    // UI
    loadingOverlay: document.getElementById('loading-overlay'),
    toastContainer: document.getElementById('toast-container')
};

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

async function initializeApp() {
    setupEventListeners();
    await checkSystemHealth();
    if (currentTab === 'dashboard') {
        await loadDashboardData();
    }
}

function setupEventListeners() {
    // Navigation
    elements.navTabs.forEach(tab => {
        tab.addEventListener('click', () => switchTab(tab.dataset.tab));
    });
    
    // Upload
    elements.uploadBox.addEventListener('click', () => elements.fileInput.click());
    elements.uploadBox.addEventListener('dragover', handleDragOver);
    elements.uploadBox.addEventListener('drop', handleDrop);
    elements.uploadBox.addEventListener('dragleave', handleDragLeave);
    elements.fileInput.addEventListener('change', handleFileSelect);
    
    // Query
    elements.queryBtn.addEventListener('click', handleQuery);
    elements.queryInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && e.ctrlKey) {
            handleQuery();
        }
    });
    
    // Dashboard
    elements.refreshStatsBtn.addEventListener('click', loadDashboardData);
    
    // Clear documents button
    if (elements.clearDocsBtn) {
        elements.clearDocsBtn.addEventListener('click', handleClearDocuments);
    }
}

// Tab Management
function switchTab(tabName) {
    currentTab = tabName;
    
    // Update nav tabs
    elements.navTabs.forEach(tab => {
        tab.classList.toggle('active', tab.dataset.tab === tabName);
    });
    
    // Update sections
    elements.sections.forEach(section => {
        section.classList.toggle('active', section.id === `${tabName}-section`);
    });
    
    // Load data for specific tabs
    if (tabName === 'dashboard') {
        loadDashboardData();
    }
}

// System Health Check
async function checkSystemHealth() {
    try {
        const response = await fetch('/health');
        const data = await response.json();
        
        updateSystemStatus(data.status, data.message);
    } catch (error) {
        updateSystemStatus('error', 'Cannot connect to server');
    }
}

function updateSystemStatus(status, message) {
    elements.statusDot.className = `status-dot ${status}`;
    elements.statusText.textContent = message;
    
    // Check every 30 seconds
    setTimeout(checkSystemHealth, 30000);
}

// File Upload Handling
function handleDragOver(e) {
    e.preventDefault();
    elements.uploadBox.classList.add('dragover');
}

function handleDragLeave(e) {
    e.preventDefault();
    elements.uploadBox.classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    elements.uploadBox.classList.remove('dragover');
    
    const files = Array.from(e.dataTransfer.files);
    handleFiles(files);
}

function handleFileSelect(e) {
    const files = Array.from(e.target.files);
    handleFiles(files);
}

function handleFiles(files) {
    const validFiles = files.filter(file => {
        const validTypes = ['.pdf', '.md', '.txt', '.docx'];
        const fileExt = '.' + file.name.split('.').pop().toLowerCase();
        return validTypes.includes(fileExt);
    });
    
    if (validFiles.length === 0) {
        showToast('error', 'No valid files selected. Please upload PDF, MD, TXT, or DOCX files.');
        return;
    }
    
    validFiles.forEach(file => {
        addToUploadQueue(file);
    });
    
    if (!isProcessing) {
        processUploadQueue();
    }
}

function addToUploadQueue(file) {
    const queueItem = {
        id: Date.now() + Math.random(),
        file: file,
        status: 'pending',
        message: 'Waiting...'
    };
    
    uploadQueue.push(queueItem);
    renderUploadQueue();
}

function renderUploadQueue() {
    if (uploadQueue.length === 0) {
        elements.uploadQueue.style.display = 'none';
        return;
    }
    
    elements.uploadQueue.style.display = 'block';
    elements.queueList.innerHTML = '';
    
    uploadQueue.forEach(item => {
        const queueItemEl = document.createElement('div');
        queueItemEl.className = `queue-item ${item.status}`;
        queueItemEl.innerHTML = `
            <div class="queue-item-info">
                <i class="fas fa-file-alt"></i>
                <div>
                    <div class="queue-item-name">${item.file.name}</div>
                    <div class="queue-item-size">${formatFileSize(item.file.size)}</div>
                </div>
            </div>
            <div class="queue-item-status ${item.status}">
                ${getStatusIcon(item.status)} ${item.message}
            </div>
        `;
        elements.queueList.appendChild(queueItemEl);
    });
}

function getStatusIcon(status) {
    switch (status) {
        case 'pending': return '<i class="fas fa-clock"></i>';
        case 'processing': return '<i class="fas fa-spinner fa-spin"></i>';
        case 'success': return '<i class="fas fa-check"></i>';
        case 'error': return '<i class="fas fa-times"></i>';
        default: return '';
    }
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

async function processUploadQueue() {
    if (isProcessing || uploadQueue.length === 0) return;
    
    isProcessing = true;
    
    while (uploadQueue.length > 0) {
        const item = uploadQueue.find(item => item.status === 'pending');
        if (!item) break;
        
        item.status = 'processing';
        item.message = 'Uploading...';
        renderUploadQueue();
        
        try {
            const formData = new FormData();
            formData.append('file', item.file);
            
            const response = await fetch('/ingest', {
                method: 'POST',
                body: formData
            });
            
            if (response.ok) {
                const result = await response.json();
                item.status = 'success';
                const chunksText = result.chunks_created === 1 ? 'chunk' : 'chunks';
                item.message = `✅ Success! ${result.chunks_created} ${chunksText} created using ${result.metadata?.storage_method || 'Feast'}`;
                
                // Show detailed success message
                if (result.chunks_created > 0) {
                    showToast('success', `📄 ${item.file.name}: ${result.chunks_created} ${chunksText} processed and stored`);
                } else {
                    showToast('warning', `⚠️ ${item.file.name}: File processed but no chunks were created`);
                }
                
                // Refresh dashboard data if user is on dashboard tab
                if (currentTab === 'dashboard') {
                    console.log('🔄 Auto-refreshing dashboard after successful upload');
                    setTimeout(() => loadDashboardData(), 1000); // Small delay to ensure data is committed
                }
            } else {
                const error = await response.json();
                item.status = 'error';
                item.message = `❌ ${error.detail || 'Upload failed'}`;
                showToast('error', `Failed to upload ${item.file.name}: ${error.detail}`);
            }
        } catch (error) {
            item.status = 'error';
            item.message = `❌ Network error: ${error.message}`;
            showToast('error', `🌐 Network error uploading ${item.file.name}: ${error.message}`);
        }
        
        renderUploadQueue();
        
        // Remove successful items after a delay
        if (item.status === 'success') {
            setTimeout(() => {
                uploadQueue = uploadQueue.filter(q => q.id !== item.id);
                renderUploadQueue();
            }, 3000);
        }
    }
    
    isProcessing = false;
    
    // Clear file input
    elements.fileInput.value = '';
}

// Query Handling
async function handleQuery() {
    const question = elements.queryInput.value.trim();
    if (!question) {
        showToast('warning', 'Please enter a question');
        return;
    }
    
    elements.queryBtn.disabled = true;
    elements.queryBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
    
    try {
        const response = await fetch('/query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                question: question,
                context_limit: parseInt(elements.contextLimit.value)
            })
        });
        
        if (response.ok) {
            const result = await response.json();
            displayQueryResults(result);
        } else {
            const error = await response.json();
            showToast('error', `Query failed: ${error.detail}`);
        }
    } catch (error) {
        showToast('error', 'Network error during query');
    } finally {
        elements.queryBtn.disabled = false;
        elements.queryBtn.innerHTML = '<i class="fas fa-search"></i> Ask Question';
    }
}

function displayQueryResults(result) {
    elements.queryResults.innerHTML = `
        <div class="query-answer">
            <h3><i class="fas fa-comment-alt"></i> Answer</h3>
            <div class="query-answer-text">${result.answer.replace(/\n/g, '<br>')}</div>
        </div>
        <div class="query-sources">
            <h4><i class="fas fa-book"></i> Sources (${result.sources.length})</h4>
            ${result.sources.map((source, index) => `
                <div class="source-item">
                    <div class="source-title">${source.metadata?.document_title || source.title || 'Unknown Document'}</div>
                    <div class="source-meta">
                        <span>Type: ${source.metadata?.document_title?.split('.').pop() || source.document_type || 'unknown'}</span>
                        <span>Chunk: ${source.metadata?.chunk_index ?? source.chunk_index ?? 'N/A'}</span>
                        <span class="relevance-score">
                            Relevance: ${(result.relevance_scores[index] * 100).toFixed(1)}%
                        </span>
                    </div>
                </div>
            `).join('')}
        </div>
    `;
    
    elements.queryResults.classList.add('show');
}

// Dashboard Data Loading
async function loadDashboardData() {
    try {
        showLoading(true);
        
        // Load stats and documents in parallel with cache-busting
        const timestamp = Date.now();
        const [statsResponse, documentsResponse] = await Promise.all([
            fetch(`/stats?t=${timestamp}`, { cache: 'no-cache' }),
            fetch(`/documents?t=${timestamp}`, { cache: 'no-cache' })
        ]);
        
        if (statsResponse.ok) {
            const stats = await statsResponse.json();
            console.log('📊 Stats API response:', stats);
            updateDashboardStats(stats);
        } else {
            console.error('❌ Stats API failed:', statsResponse.status, statsResponse.statusText);
            showToast('error', 'Failed to load dashboard statistics');
        }
        
        if (documentsResponse.ok) {
            const documents = await documentsResponse.json();
            console.log('📚 Documents API response:', documents);
            updateDocumentsList(documents);
        } else {
            console.error('❌ Documents API failed:', documentsResponse.status, documentsResponse.statusText);
            showToast('error', 'Failed to load documents list');
        }
        
    } catch (error) {
        console.error('❌ Dashboard loading error:', error);
        showToast('error', `Failed to load dashboard data: ${error.message}`);
    } finally {
        showLoading(false);
    }
}

function updateDashboardStats(stats) {
    elements.docCount.textContent = stats.vector_store_stats?.document_count || '0';
    elements.llmModel.textContent = stats.llm_model || 'N/A';
    elements.embeddingModel.textContent = stats.embedding_model || 'N/A';
    elements.pipelineStatus.textContent = stats.pipeline_status || 'Unknown';
    
    // Update status to reflect Feast + Milvus-lite architecture
    const isConnected = stats.vector_store_stats && stats.pipeline_status === 'ready';
    elements.dbStatus.textContent = isConnected ? 'Feast + Milvus-lite Connected' : 'Disconnected';
}

function updateDocumentsList(documents) {
    if (!documents || !documents.documents || documents.documents.length === 0) {
        elements.documentsContainer.innerHTML = '<div class="loading">No documents found</div>';
        return;
    }
    
    elements.documentsContainer.innerHTML = documents.documents.map(doc => `
        <div class="document-item">
            <div class="document-info">
                <div class="document-name">📄 ${doc.title || 'Unknown Document'}</div>
                <div class="document-meta">
                    <span class="meta-badge">Type: ${doc.document_type}</span> • 
                    <span class="meta-badge chunks">📊 ${doc.chunk_count || 'N/A'} chunks</span> • 
                    <span class="meta-badge">📅 ${doc.created_at ? new Date(doc.created_at).toLocaleDateString() : 'Unknown'}</span>
                </div>
            </div>
        </div>
    `).join('');
}

// Clear Documents
async function handleClearDocuments() {
    if (!confirm('Are you sure you want to clear all documents? This action cannot be undone.')) {
        return;
    }
    
    try {
        showLoading(true);
        
        const response = await fetch('/documents', {
            method: 'DELETE'
        });
        
        if (response.ok) {
            const result = await response.json();
            showToast('success', 'All documents cleared successfully');
            await loadDashboardData();
        } else {
            const error = await response.json();
            showToast('error', `Failed to clear documents: ${error.detail}`);
        }
    } catch (error) {
        showToast('error', 'Network error clearing documents');
    } finally {
        showLoading(false);
    }
}

// Utility Functions
function showLoading(show) {
    elements.loadingOverlay.style.display = show ? 'flex' : 'none';
}

function showToast(type, message) {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `
        <div class="toast-content">
            <i class="toast-icon ${type} ${getToastIcon(type)}"></i>
            <div class="toast-message">${message}</div>
        </div>
    `;
    
    elements.toastContainer.appendChild(toast);
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        toast.remove();
    }, 5000);
}

function getToastIcon(type) {
    switch (type) {
        case 'success': return 'fas fa-check-circle';
        case 'error': return 'fas fa-exclamation-circle';
        case 'warning': return 'fas fa-exclamation-triangle';
        default: return 'fas fa-info-circle';
    }
}

// Keyboard shortcuts
document.addEventListener('keydown', function(e) {
    // Ctrl+U for upload tab
    if (e.ctrlKey && e.key === 'u') {
        e.preventDefault();
        switchTab('upload');
    }
    
    // Ctrl+Q for query tab
    if (e.ctrlKey && e.key === 'q') {
        e.preventDefault();
        switchTab('query');
    }
    
    // Ctrl+D for dashboard tab
    if (e.ctrlKey && e.key === 'd') {
        e.preventDefault();
        switchTab('dashboard');
    }
});

// Auto-refresh dashboard when visible
setInterval(() => {
    if (currentTab === 'dashboard') {
        loadDashboardData();
    }
}, 30000); // Refresh every 30 seconds 