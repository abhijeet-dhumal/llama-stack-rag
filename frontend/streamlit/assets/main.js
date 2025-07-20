// RAG LlamaStack Application JavaScript

// Enhanced animations and interactions
document.addEventListener('DOMContentLoaded', function() {
    
    // Add fade-in animation to new elements
    function addFadeInAnimation() {
        const newElements = document.querySelectorAll('[data-testid="stMarkdownContainer"]:not(.animated)');
        newElements.forEach(element => {
            element.classList.add('fade-in', 'animated');
        });
    }

    // Progress bar enhancement
    function enhanceProgressBars() {
        const progressBars = document.querySelectorAll('.stProgress');
        progressBars.forEach(bar => {
            if (!bar.classList.contains('enhanced')) {
                bar.classList.add('enhanced');
                
                // Add pulse animation for active progress bars
                const progressFill = bar.querySelector('div > div > div');
                if (progressFill) {
                    const observer = new MutationObserver(() => {
                        if (progressFill.style.width !== '0%' && progressFill.style.width !== '100%') {
                            progressFill.style.animation = 'pulse 1.5s infinite';
                        } else {
                            progressFill.style.animation = 'none';
                        }
                    });
                    observer.observe(progressFill, { attributes: true, attributeFilter: ['style'] });
                }
            }
        });
    }

    // Button hover effects
    function enhanceButtons() {
        const buttons = document.querySelectorAll('.stButton > button:not(.enhanced)');
        buttons.forEach(button => {
            button.classList.add('enhanced');
            
            button.addEventListener('mouseenter', function() {
                this.style.transform = 'translateY(-2px) scale(1.02)';
            });
            
            button.addEventListener('mouseleave', function() {
                this.style.transform = 'translateY(0) scale(1)';
            });
            
            button.addEventListener('mousedown', function() {
                this.style.transform = 'translateY(1px) scale(0.98)';
            });
            
            button.addEventListener('mouseup', function() {
                this.style.transform = 'translateY(-2px) scale(1.02)';
            });
        });
    }

    // File upload drag and drop enhancement
    function enhanceFileUpload() {
        const uploadAreas = document.querySelectorAll('[data-testid="stFileUploader"]');
        uploadAreas.forEach(area => {
            if (!area.classList.contains('enhanced')) {
                area.classList.add('enhanced', 'upload-area');
                
                area.addEventListener('dragover', function(e) {
                    e.preventDefault();
                    this.style.borderColor = '#764ba2';
                    this.style.background = 'linear-gradient(135deg, rgba(102,126,234,0.3) 0%, rgba(118,75,162,0.3) 100%)';
                });
                
                area.addEventListener('dragleave', function(e) {
                    e.preventDefault();
                    this.style.borderColor = '#667eea';
                    this.style.background = 'linear-gradient(135deg, rgba(102,126,234,0.1) 0%, rgba(118,75,162,0.1) 100%)';
                });
                
                area.addEventListener('drop', function(e) {
                    this.style.borderColor = '#667eea';
                    this.style.background = 'linear-gradient(135deg, rgba(102,126,234,0.1) 0%, rgba(118,75,162,0.1) 100%)';
                });
            }
        });
    }

    // Sidebar collapse animation
    function enhanceSidebar() {
        const sidebar = document.querySelector('[data-testid="stSidebar"]');
        if (sidebar && !sidebar.classList.contains('enhanced')) {
            sidebar.classList.add('enhanced');
            
            // Add smooth transitions
            sidebar.style.transition = 'all 0.3s ease';
            
            // Observe sidebar state changes
            const observer = new MutationObserver(() => {
                if (sidebar.getAttribute('aria-expanded') === 'false') {
                    sidebar.style.transform = 'translateX(-100%)';
                } else {
                    sidebar.style.transform = 'translateX(0)';
                }
            });
            
            observer.observe(sidebar, { attributes: true, attributeFilter: ['aria-expanded'] });
        }
    }

    // Auto-scroll to bottom for chat-like interfaces
    function autoScrollChat() {
        const chatContainers = document.querySelectorAll('.chat-container');
        chatContainers.forEach(container => {
            container.scrollTop = container.scrollHeight;
        });
    }

    // Toast notification system
    function createToast(message, type = 'info', duration = 3000) {
        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        toast.innerHTML = `
            <div class="toast-content">
                <span class="toast-icon">${getToastIcon(type)}</span>
                <span class="toast-message">${message}</span>
                <button class="toast-close" onclick="this.parentElement.parentElement.remove()">×</button>
            </div>
        `;
        
        toast.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: ${getToastColor(type)};
            color: white;
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            z-index: 10000;
            animation: slideInRight 0.3s ease;
            max-width: 300px;
        `;
        
        document.body.appendChild(toast);
        
        setTimeout(() => {
            if (toast.parentElement) {
                toast.style.animation = 'slideOutRight 0.3s ease';
                setTimeout(() => toast.remove(), 300);
            }
        }, duration);
    }

    function getToastIcon(type) {
        const icons = {
            success: '✅',
            error: '❌', 
            warning: '⚠️',
            info: 'ℹ️'
        };
        return icons[type] || icons.info;
    }

    function getToastColor(type) {
        const colors = {
            success: 'linear-gradient(45deg, #56ab2f, #a8e6cf)',
            error: 'linear-gradient(45deg, #ff416c, #ff4b2b)',
            warning: 'linear-gradient(45deg, #f7971e, #ffd200)',
            info: 'linear-gradient(45deg, #667eea, #764ba2)'
        };
        return colors[type] || colors.info;
    }

    // Enhanced metrics animation
    function animateMetrics() {
        const metrics = document.querySelectorAll('.metric-value:not(.animated)');
        metrics.forEach(metric => {
            metric.classList.add('animated');
            const finalValue = parseInt(metric.textContent);
            let currentValue = 0;
            const increment = finalValue / 50;
            const timer = setInterval(() => {
                currentValue += increment;
                if (currentValue >= finalValue) {
                    currentValue = finalValue;
                    clearInterval(timer);
                }
                metric.textContent = Math.floor(currentValue);
            }, 20);
        });
    }

    // Performance monitoring
    function monitorPerformance() {
        if (performance.mark && performance.measure) {
            performance.mark('ui-enhancement-start');
            
            // Add all enhancements
            addFadeInAnimation();
            enhanceProgressBars();
            enhanceButtons();
            enhanceFileUpload();
            enhanceSidebar();
            autoScrollChat();
            animateMetrics();
            
            performance.mark('ui-enhancement-end');
            performance.measure('ui-enhancement', 'ui-enhancement-start', 'ui-enhancement-end');
            
            const measure = performance.getEntriesByName('ui-enhancement')[0];
            console.log(`UI enhancements loaded in ${measure.duration.toFixed(2)}ms`);
        }
    }

    // Initialize all enhancements
    monitorPerformance();

    // Re-run enhancements when Streamlit updates the DOM
    const observer = new MutationObserver(() => {
        addFadeInAnimation();
        enhanceProgressBars();
        enhanceButtons();
        enhanceFileUpload();
        enhanceSidebar();
        autoScrollChat();
        animateMetrics();
    });

    observer.observe(document.body, {
        childList: true,
        subtree: true
    });

    // Expose functions globally for Streamlit components
    window.ragApp = {
        createToast,
        animateMetrics,
        enhanceProgressBars
    };
});

// CSS animations (injected via JavaScript)
const style = document.createElement('style');
style.textContent = `
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    
    @keyframes slideInRight {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes slideOutRight {
        from { transform: translateX(0); opacity: 1; }
        to { transform: translateX(100%); opacity: 0; }
    }
    
    .toast-content {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .toast-close {
        background: none;
        border: none;
        color: white;
        font-size: 1.2rem;
        cursor: pointer;
        margin-left: auto;
    }
`;
document.head.appendChild(style); 