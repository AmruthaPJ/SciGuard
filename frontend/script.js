const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const loading = document.getElementById('loading');
const resultSection = document.getElementById('result-section');
const statusText = document.getElementById('status-text');

// --- 1. EVENT LISTENERS ---

dropZone.addEventListener('click', () => fileInput.click());

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('drag-over');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    if (e.dataTransfer.files.length > 0) {
        handleFileUpload(e.dataTransfer.files[0]);
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFileUpload(e.target.files[0]);
    }
});

// --- 2. CORE LOGIC ---

async function handleFileUpload(file) {
    // UI Transitions
    dropZone.classList.add('hidden');
    document.querySelector('.hero').classList.add('hidden');
    loading.classList.remove('hidden');
    
    const formData = new FormData();
    formData.append('image', file);

    // Update Status messages for flavor
    const stages = ["EXTRACTING PIXELS...", "RUNNING M1 SPATIAL STREAM...", "RUNNING M2 FORENSIC STREAM...", "GATING FUSION MODULE..."];
    let i = 0;
    const interval = setInterval(() => {
        statusText.innerText = stages[i % stages.length];
        i++;
    }, 800);

    try {
        const API_BASE = window.location.protocol === 'file:' ? 'http://localhost:5001' : '';
        const response = await fetch(`${API_BASE}/analyze`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error("Analysis Failed");

        const data = await response.json();
        clearInterval(interval);
        showResults(data, file);
    } catch (err) {
        console.error(err);
        alert("Server connection failed. Ensure app.py is running on port 5001.");
        window.location.reload();
    }
}

function showResults(data, file) {
    loading.classList.add('hidden');
    resultSection.classList.remove('hidden');

    // Set Images
    const reader = new FileReader();
    reader.onload = (e) => {
        document.getElementById('source-img').src = e.target.result;
    };
    reader.readAsDataURL(file);

    document.getElementById('overlay-img').src = data.mask_overlay;

    // Set Verdict
    const verdictEl = document.getElementById('verdict-label');
    verdictEl.innerText = data.verdict === "High" ? "HIGH RISK DETECTED" : "INTEGRITY VERIFIED";
    verdictEl.style.color = data.verdict === "High" ? "#ff3366" : "#00ffcc";
    
    document.getElementById('confidence-score').innerText = `${data.confidence}%`;

    // Set Stats
    document.getElementById('m1-bar').style.width = `${data.stats.spatial_trust}%`;
    document.getElementById('m2-bar').style.width = `${data.stats.forensic_trust}%`;

    // Update progress bar colors based on trust
    document.getElementById('m1-bar').style.background = data.stats.spatial_trust > 50 ? "#00ffcc" : "#8892b0";
    document.getElementById('m2-bar').style.background = data.stats.forensic_trust > 50 ? "#00ffcc" : "#8892b0";
}
