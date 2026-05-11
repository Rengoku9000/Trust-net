document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const uploadSection = document.getElementById('upload-section');
    const loadingState = document.getElementById('loading-state');
    const dashboard = document.getElementById('dashboard');
    const btnReset = document.getElementById('btn-reset');

    // Prevent default drag behaviors
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
        document.body.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    // Highlight drop zone when item is dragged over it
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, unhighlight, false);
    });

    function highlight(e) {
        dropZone.classList.add('dragover');
    }

    function unhighlight(e) {
        dropZone.classList.remove('dragover');
    }

    // Handle dropped files
    dropZone.addEventListener('drop', handleDrop, false);

    function handleDrop(e) {
        let dt = e.dataTransfer;
        let files = dt.files;
        if (files.length) {
            handleFiles(files[0]);
        }
    }

    // Handle file input change
    fileInput.addEventListener('change', function() {
        if (this.files.length) {
            handleFiles(this.files[0]);
        }
    });

    // Handle the selected file
    function handleFiles(file) {
        uploadFile(file);
    }

    // Upload file and get analysis
    async function uploadFile(file) {
        // Show loading state
        dropZone.classList.add('hidden');
        loadingState.classList.remove('hidden');

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/analyze', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Server error: ${response.statusText}`);
            }

            const data = await response.json();
            
            // Artificial delay for cinematic effect (simulating deep analysis)
            setTimeout(() => {
                displayResults(data);
            }, 800);

        } catch (error) {
            console.error('Error uploading file:', error);
            alert('An error occurred during analysis. Please try again.');
            resetApp();
        }
    }

    function getRiskColor(score) {
        if (score >= 70) return 'var(--risk-high)';
        if (score >= 40) return 'var(--risk-medium)';
        return 'var(--risk-low)';
    }

    function updateScoreRing(score) {
        const ring = document.getElementById('score-ring-progress');
        const scoreText = document.getElementById('unified-score');
        const explanation = document.getElementById('score-explanation');
        
        // 440 is the max dashoffset (empty ring)
        const offset = 440 - (score / 100) * 440;
        const color = getRiskColor(score);
        
        // Animate counter
        let current = 0;
        const target = score;
        const step = target / 30; // 30 frames
        
        const counter = setInterval(() => {
            current += step;
            if (current >= target) {
                current = target;
                clearInterval(counter);
            }
            scoreText.textContent = Math.round(current);
            scoreText.style.color = color;
        }, 30);

        // Update SVG styles
        requestAnimationFrame(() => {
            ring.style.strokeDashoffset = offset;
            ring.style.stroke = color;
        });
    }

    function updateLayer(layerId, data) {
        const scoreElement = document.getElementById(`score-${layerId}`);
        const progressElement = document.getElementById(`progress-${layerId}`);
        const msgElement = document.getElementById(`msg-${layerId}`);
        
        const score = data.score !== undefined ? data.score : data; // Handle both direct score and object
        const color = getRiskColor(score);
        
        // Update score text
        scoreElement.textContent = score;
        scoreElement.style.color = color;
        
        // Animate progress bar
        requestAnimationFrame(() => {
            progressElement.style.width = `${score}%`;
            progressElement.style.backgroundColor = color;
        });

        // Update message if exists
        if (msgElement && data.message) {
            msgElement.textContent = data.message;
        } else if (msgElement) {
            msgElement.textContent = score >= 70 ? "High risk indicators found" : 
                                     score >= 40 ? "Moderate anomalies detected" : 
                                     "No significant issues detected";
        }
    }

    function formatBytes(bytes, decimals = 2) {
        if (!+bytes) return '0 Bytes';
        const k = 1024;
        const dm = decimals < 0 ? 0 : decimals;
        const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return `${parseFloat((bytes / Math.pow(k, i)).toFixed(dm))} ${sizes[i]}`;
    }

    function displayResults(data) {
        // Hide upload section, show dashboard
        uploadSection.classList.add('hidden');
        dashboard.classList.remove('hidden');

        // Update file info
        document.getElementById('filename-display').textContent = data.filename;
        document.getElementById('filesize-display').textContent = formatBytes(data.file_size_bytes);

        // Update main score explanation
        const explanation = document.getElementById('score-explanation');
        explanation.textContent = data.explanation;
        
        // Set dynamic background color based on risk
        const riskColor = getRiskColor(data.unified_risk_score);
        explanation.style.backgroundColor = `rgba(${
            data.unified_risk_score >= 70 ? '239, 68, 68' : 
            data.unified_risk_score >= 40 ? '245, 158, 11' : '16, 185, 129'
        }, 0.1)`;
        explanation.style.color = riskColor;
        explanation.style.border = `1px solid ${riskColor}`;

        // Animate all scores
        updateScoreRing(data.unified_risk_score);
        
        const layers = data.layer_scores;
        
        // Delay layer animations slightly for cascading effect
        setTimeout(() => updateLayer('ela', layers.ela), 100);
        setTimeout(() => updateLayer('benfords', layers.benfords_law), 200);
        setTimeout(() => updateLayer('metadata', layers.metadata), 300);
        setTimeout(() => updateLayer('nlp', layers.nlp), 400);
        setTimeout(() => updateLayer('gnn', layers.graph), 500);
    }

    function resetApp() {
        // Hide dashboard, show upload section
        dashboard.classList.add('hidden');
        uploadSection.classList.remove('hidden');
        dropZone.classList.remove('hidden');
        loadingState.classList.add('hidden');

        // Reset inputs
        fileInput.value = '';

        // Reset progress bars and rings immediately
        document.getElementById('score-ring-progress').style.strokeDashoffset = 440;
        document.getElementById('score-ring-progress').style.stroke = 'var(--accent-primary)';
        document.getElementById('unified-score').textContent = '0';
        document.getElementById('unified-score').style.color = 'var(--text-main)';
        
        const layers = ['ela', 'benfords', 'metadata', 'nlp', 'gnn'];
        layers.forEach(layer => {
            const progressElement = document.getElementById(`progress-${layer}`);
            if (progressElement) {
                progressElement.style.width = '0%';
            }
        });
    }

    // Reset button listener
    btnReset.addEventListener('click', resetApp);
});
