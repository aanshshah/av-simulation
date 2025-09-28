/**
 * Main JavaScript for AV Simulation GitHub Pages
 */

document.addEventListener('DOMContentLoaded', function() {
    // Initialize components
    initializeNavigation();
    initializeCodeBlocks();
    initializeImageLoading();
    initializeScrollToTop();
    initializeAnalytics();
});

/**
 * Navigation enhancements
 */
function initializeNavigation() {
    // Highlight current page in navigation
    const currentPath = window.location.pathname;
    const navLinks = document.querySelectorAll('.nav-links a');

    navLinks.forEach(link => {
        const href = link.getAttribute('href');
        if (href && currentPath.includes(href)) {
            link.classList.add('active');
        }
    });

    // Mobile navigation toggle
    const navToggle = document.querySelector('.nav-toggle');
    const navLinks = document.querySelector('.nav-links');

    if (navToggle && navLinks) {
        navToggle.addEventListener('click', function() {
            navLinks.classList.toggle('active');
        });
    }

    // Smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
}

/**
 * Code block enhancements
 */
function initializeCodeBlocks() {
    // Add copy buttons to code blocks
    document.querySelectorAll('pre').forEach(pre => {
        const code = pre.querySelector('code');
        if (code) {
            const copyBtn = document.createElement('button');
            copyBtn.className = 'copy-btn';
            copyBtn.innerHTML = '📋 Copy';
            copyBtn.title = 'Copy code to clipboard';

            copyBtn.addEventListener('click', async function() {
                try {
                    await navigator.clipboard.writeText(code.textContent);
                    copyBtn.innerHTML = '✅ Copied!';
                    setTimeout(() => {
                        copyBtn.innerHTML = '📋 Copy';
                    }, 2000);
                } catch (err) {
                    console.error('Failed to copy code:', err);
                    copyBtn.innerHTML = '❌ Failed';
                    setTimeout(() => {
                        copyBtn.innerHTML = '📋 Copy';
                    }, 2000);
                }
            });

            pre.style.position = 'relative';
            pre.appendChild(copyBtn);
        }
    });

    // Line number highlighting for code blocks
    document.querySelectorAll('pre code').forEach(code => {
        const lines = code.innerHTML.split('\n');
        if (lines.length > 1) {
            code.innerHTML = lines.map((line, index) => {
                return `<span class="line" data-line="${index + 1}">${line}</span>`;
            }).join('\n');
        }
    });
}

/**
 * Image loading enhancements
 */
function initializeImageLoading() {
    // Lazy loading for images
    if ('IntersectionObserver' in window) {
        const imageObserver = new IntersectionObserver((entries, observer) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const img = entry.target;
                    img.src = img.dataset.src;
                    img.classList.remove('lazy');
                    imageObserver.unobserve(img);
                }
            });
        });

        document.querySelectorAll('img[data-src]').forEach(img => {
            imageObserver.observe(img);
        });
    }

    // Image modal for larger viewing
    document.querySelectorAll('img').forEach(img => {
        img.addEventListener('click', function() {
            if (this.naturalWidth > this.clientWidth) {
                openImageModal(this.src, this.alt);
            }
        });
    });
}

/**
 * Scroll to top functionality
 */
function initializeScrollToTop() {
    // Create scroll to top button
    const scrollBtn = document.createElement('button');
    scrollBtn.innerHTML = '↑';
    scrollBtn.className = 'scroll-top-btn';
    scrollBtn.title = 'Scroll to top';
    scrollBtn.style.display = 'none';
    document.body.appendChild(scrollBtn);

    // Show/hide scroll button based on scroll position
    let isVisible = false;
    window.addEventListener('scroll', function() {
        const shouldShow = window.scrollY > 300;
        if (shouldShow && !isVisible) {
            scrollBtn.style.display = 'block';
            scrollBtn.style.opacity = '1';
            isVisible = true;
        } else if (!shouldShow && isVisible) {
            scrollBtn.style.opacity = '0';
            setTimeout(() => {
                if (!isVisible) scrollBtn.style.display = 'none';
            }, 300);
            isVisible = false;
        }
    });

    // Scroll to top when clicked
    scrollBtn.addEventListener('click', function() {
        window.scrollTo({
            top: 0,
            behavior: 'smooth'
        });
    });
}

/**
 * Analytics and tracking
 */
function initializeAnalytics() {
    // Track external links
    document.querySelectorAll('a[href^="http"]').forEach(link => {
        link.addEventListener('click', function() {
            if (typeof gtag !== 'undefined') {
                gtag('event', 'click', {
                    event_category: 'external_link',
                    event_label: this.href
                });
            }
        });
    });

    // Track code copy events
    document.addEventListener('click', function(e) {
        if (e.target.classList.contains('copy-btn')) {
            if (typeof gtag !== 'undefined') {
                gtag('event', 'code_copy', {
                    event_category: 'engagement'
                });
            }
        }
    });

    // Track page views
    if (typeof gtag !== 'undefined') {
        gtag('event', 'page_view', {
            page_title: document.title,
            page_location: window.location.href
        });
    }
}

/**
 * Image modal functionality
 */
function openImageModal(src, alt) {
    // Create modal elements
    const modal = document.createElement('div');
    modal.className = 'image-modal';
    modal.innerHTML = `
        <div class="modal-backdrop"></div>
        <div class="modal-content">
            <button class="modal-close">&times;</button>
            <img src="${src}" alt="${alt}">
            <div class="modal-caption">${alt}</div>
        </div>
    `;

    // Add to page
    document.body.appendChild(modal);
    document.body.style.overflow = 'hidden';

    // Close modal events
    const closeModal = () => {
        document.body.removeChild(modal);
        document.body.style.overflow = '';
    };

    modal.querySelector('.modal-close').addEventListener('click', closeModal);
    modal.querySelector('.modal-backdrop').addEventListener('click', closeModal);

    // Close on escape key
    const escapeHandler = (e) => {
        if (e.key === 'Escape') {
            closeModal();
            document.removeEventListener('keydown', escapeHandler);
        }
    };
    document.addEventListener('keydown', escapeHandler);
}

/**
 * Theme switching functionality (for future enhancement)
 */
function initializeThemeSwitch() {
    const themeSwitch = document.querySelector('.theme-switch');
    if (themeSwitch) {
        const currentTheme = localStorage.getItem('theme') || 'light';
        document.documentElement.setAttribute('data-theme', currentTheme);

        themeSwitch.addEventListener('click', function() {
            const newTheme = document.documentElement.getAttribute('data-theme') === 'light' ? 'dark' : 'light';
            document.documentElement.setAttribute('data-theme', newTheme);
            localStorage.setItem('theme', newTheme);
        });
    }
}

/**
 * Search functionality (for future enhancement)
 */
function initializeSearch() {
    const searchInput = document.querySelector('.search-input');
    if (searchInput) {
        let searchTimeout;
        searchInput.addEventListener('input', function() {
            clearTimeout(searchTimeout);
            searchTimeout = setTimeout(() => {
                performSearch(this.value);
            }, 300);
        });
    }
}

function performSearch(query) {
    // Implement search functionality
    console.log('Searching for:', query);
}

/**
 * Progress indicator for long pages
 */
function initializeReadingProgress() {
    const progressBar = document.createElement('div');
    progressBar.className = 'reading-progress';
    document.body.appendChild(progressBar);

    window.addEventListener('scroll', function() {
        const scrollPercent = (window.scrollY / (document.body.scrollHeight - window.innerHeight)) * 100;
        progressBar.style.width = Math.min(scrollPercent, 100) + '%';
    });
}

/**
 * Table of contents generation
 */
function generateTableOfContents() {
    const headings = document.querySelectorAll('h1, h2, h3, h4, h5, h6');
    if (headings.length > 3) {
        const toc = document.createElement('nav');
        toc.className = 'table-of-contents';
        toc.innerHTML = '<h3>Table of Contents</h3><ul></ul>';

        const tocList = toc.querySelector('ul');
        headings.forEach((heading, index) => {
            const id = heading.id || `heading-${index}`;
            heading.id = id;

            const li = document.createElement('li');
            li.className = `toc-${heading.tagName.toLowerCase()}`;
            li.innerHTML = `<a href="#${id}">${heading.textContent}</a>`;
            tocList.appendChild(li);
        });

        // Insert TOC after the first heading
        const firstHeading = document.querySelector('h1');
        if (firstHeading && firstHeading.nextElementSibling) {
            firstHeading.parentNode.insertBefore(toc, firstHeading.nextElementSibling);
        }
    }
}

// Export functions for potential external use
window.AVSimulation = {
    initializeNavigation,
    initializeCodeBlocks,
    openImageModal,
    generateTableOfContents
};