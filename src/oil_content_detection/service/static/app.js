const form = document.querySelector('#predict-form');
const submitButton = document.querySelector('#submit-button');
const statusEl = document.querySelector('#status');
const healthEl = document.querySelector('#health');
const resultCard = document.querySelector('#result-card');
const visualCard = document.querySelector('#visual-card');
const jsonCard = document.querySelector('#json-card');
const visualGrid = document.querySelector('#visual-grid');
const jsonOutput = document.querySelector('#json-output');

function setText(selector, value) {
  const element = document.querySelector(selector);
  if (element) element.textContent = value;
}

function formatPercent(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '--';
  return `${(Number(value) * 100).toFixed(2)}%`;
}

function formatNumber(value, digits = 4) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '--';
  return Number(value).toFixed(digits);
}

function setStatus(message, kind = '') {
  statusEl.textContent = message;
  statusEl.className = `status ${kind}`.trim();
}

async function refreshHealth() {
  try {
    const response = await fetch('/api/health');
    const payload = await response.json();
    if (!response.ok || payload.status !== 'ok') {
      healthEl.textContent = `模型未就绪：${payload.error || 'unknown error'}`;
      healthEl.className = 'health error';
      return;
    }
    healthEl.textContent = `模型已加载：${payload.model_version || 'unknown'}`;
    healthEl.className = 'health ok';
  } catch (error) {
    healthEl.textContent = `服务不可用：${error}`;
    healthEl.className = 'health error';
  }
}

function makeVisual(title, url) {
  const item = document.createElement('article');
  item.className = 'visual-item';
  const heading = document.createElement('h3');
  heading.textContent = title;
  item.appendChild(heading);
  if (url) {
    const link = document.createElement('a');
    link.href = url;
    link.target = '_blank';
    link.rel = 'noreferrer';
    const image = document.createElement('img');
    image.src = url;
    image.alt = title;
    link.appendChild(image);
    item.appendChild(link);
  } else {
    const empty = document.createElement('p');
    empty.className = 'empty';
    empty.textContent = '未生成';
    item.appendChild(empty);
  }
  return item;
}

function renderWarnings(warnings) {
  const warningsEl = document.querySelector('#warnings');
  if (!warningsEl) return;
  warningsEl.replaceChildren();
  if (!warnings || warnings.length === 0) {
    warningsEl.classList.add('hidden');
    return;
  }
  const title = document.createElement('strong');
  title.textContent = '警告：';
  warningsEl.appendChild(title);
  const list = document.createElement('ul');
  warnings.forEach((warning) => {
    const item = document.createElement('li');
    item.textContent = warning;
    list.appendChild(item);
  });
  warningsEl.appendChild(list);
  warningsEl.classList.remove('hidden');
}

function renderResult(payload) {
  const prediction = payload.prediction || {};
  const roi = prediction.roi || {};
  const qc = prediction.qc || {};
  const visualizations = payload.visualizations || {};

  setText('#prediction-value', formatNumber(prediction.prediction, 4));
  setText('#prediction-unit', prediction.unit || 'ml/100g');
  setText('#qc-status', qc.status || '--');
  setText('#model-version', prediction.model_version || '--');
  setText('#job-id', payload.job_id || '--');
  setText('#coverage-ratio', formatPercent(roi.coverage_ratio));
  setText('#pepper-pixels', roi.pepper_pixels ?? '--');
  setText('#valid-pixel-ratio', formatPercent(roi.valid_pixel_ratio));
  renderWarnings(qc.warnings || []);

  visualGrid.replaceChildren(
    makeVisual('上传 RGB 图片', visualizations.image_url),
    makeVisual('ROI Mask', visualizations.mask_url),
    makeVisual('ROI Overlay', visualizations.overlay_url),
    makeVisual('含油率模型响应热力图', visualizations.heatmap_url),
    makeVisual('热力图叠加图', visualizations.heatmap_overlay_url),
  );

  jsonOutput.textContent = JSON.stringify(payload, null, 2);
  resultCard.classList.remove('hidden');
  visualCard.classList.remove('hidden');
  jsonCard.classList.remove('hidden');
}

function renderError(payload) {
  const error = payload.error || {};
  resultCard.classList.add('hidden');
  visualCard.classList.add('hidden');
  jsonCard.classList.remove('hidden');
  jsonOutput.textContent = JSON.stringify(payload, null, 2);
  setStatus(`检测失败：${error.code || 'ERROR'} - ${error.message || '未知错误'}`, 'error');
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  setStatus('正在上传并检测，请稍候…', 'busy');
  submitButton.disabled = true;
  try {
    const formData = new FormData(form);
    const heatmapInput = form.querySelector('input[name="generate_heatmap"]');
    formData.set('generate_heatmap', heatmapInput && heatmapInput.checked ? 'true' : 'false');
    const response = await fetch('/api/predict', { method: 'POST', body: formData });
    const payload = await response.json();
    if (!response.ok) {
      renderError(payload);
      return;
    }
    setStatus('检测完成。', 'ok');
    renderResult(payload);
  } catch (error) {
    setStatus(`请求失败：${error}`, 'error');
  } finally {
    submitButton.disabled = false;
  }
});

refreshHealth();
