let data, thresholdIndex = 0, caseIndex = 0, imageView = 'pred', selectedField = null;
const $ = (id) => document.getElementById(id);
const pct = (n, places = 1) => n == null ? '—' : `${(100 * n).toFixed(places)}%`;
const esc = (text) => String(text).replace(/[&<>"']/g, (ch) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));

function line(points, style) { return `<polyline class="${style}" points="${points.join(' ')}"/>`; }
function grid(width, height, left, top, bottom, ticks) {
  let output = '';
  for (const fraction of ticks) {
    const y = top + (height - top - bottom) * (1 - fraction);
    output += `<line x1="${left}" x2="${width - 20}" y1="${y}" y2="${y}" class="chart-grid"/><text x="8" y="${y + 4}" class="chart-axis">${pct(fraction, 0)}</text>`;
  }
  return output;
}

function operatingChart() {
  const values = data.operating_curve, width = 760, height = 320, left = 60, right = 27, top = 21, bottom = 49;
  const x = (i) => left + (width - left - right) * i / (values.length - 1);
  const y = (n) => top + (height - top - bottom) * (1 - n);
  let output = grid(width, height, left, top, bottom, [0, .25, .5, .75, 1]);
  output += line(values.map((r, i) => `${x(i)},${y(r.val_crop_error)}`), 'op-line op-val');
  output += line(values.map((r, i) => `${x(i)},${y(r.test_crop_error)}`), 'op-line op-test');
  output += line(values.map((r, i) => `${x(i)},${y(r.test_weed_recall)}`), 'op-line op-recall');
  const chosen = values[thresholdIndex];
  output += `<line x1="${x(thresholdIndex)}" x2="${x(thresholdIndex)}" y1="${top}" y2="${height-bottom}" class="op-marker"/>`;
  for (const [field, cls] of [['val_crop_error','op-val'],['test_crop_error','op-test'],['test_weed_recall','op-recall']]) output += `<circle cx="${x(thresholdIndex)}" cy="${y(chosen[field])}" r="6" class="op-dot ${cls}"/>`;
  for (const index of [0, 4, 8, 12, 16, 20, 24, values.length - 1]) output += `<text x="${x(index)}" y="${height-16}" text-anchor="middle" class="chart-axis">${values[index].threshold.toFixed(2)}</text>`;
  $('operating-chart').innerHTML = output;
  $('inspect-threshold').textContent = chosen.threshold.toFixed(3);
  $('val-crop-error').textContent = pct(chosen.val_crop_error);
  $('test-crop-error').textContent = pct(chosen.test_crop_error);
  $('inspect-weed-recall').textContent = pct(chosen.test_weed_recall);
  const defaultPoint = Math.abs(chosen.threshold - data.threshold) < .001;
  $('threshold-note').textContent = defaultPoint ? 'Selected on validation images. The test result breaches the illustrative crop-risk gate.' : 'Inspection only. This point was not the preselected threshold; the overlap experiment and release gate stay fixed.';
}

function overlapChart() {
  const values = data.sweep, width = 900, height = 320, left = 70, right = 30, top = 20, bottom = 67;
  const x = (i) => left + (width - left - right) * (i + .5) / values.length;
  const y = (n) => top + (height - top - bottom) * (1 - n);
  let output = grid(width, height, left, top, bottom, [0, .25, .5, .75, 1]);
  values.forEach((row, i) => {
    output += `<rect class="stress-bar" x="${x(i)-34}" y="${y(row.crop_false_weed_rate)}" width="68" height="${height-bottom-y(row.crop_false_weed_rate)}" rx="7"/>`;
    output += `<text class="chart-axis" x="${x(i)}" y="${height-38}" text-anchor="middle">${pct(row.target,0)} target</text><text class="chart-axis" x="${x(i)}" y="${height-20}" text-anchor="middle">${pct(row.actual_mean,0)} actual</text>`;
  });
  output += line(values.map((r,i) => `${x(i)},${y(r.weed_pixel_recall)}`), 'stress-line');
  values.forEach((row,i) => output += `<circle class="stress-dot" cx="${x(i)}" cy="${y(row.weed_pixel_recall)}" r="5"/><text class="chart-value" x="${x(i)}" y="${y(row.weed_pixel_recall)-12}" text-anchor="middle">${pct(row.weed_pixel_recall,0)}</text>`);
  $('overlap-chart').innerHTML = output;
  const atForty = values.find((r) => r.target === .4);
  $('overlap-finding').textContent = `${pct(atForty.crop_false_weed_rate,0)} crop activation`;
  $('overlap-detail').textContent = `At ${pct(atForty.actual_mean,0)} measured overlap, ${pct(atForty.crop_false_weed_rate,0)} of inserted crop pixels crossed the weed threshold. Composed-image stress result; no crop damage was measured.`;
}

function cases() {
  let rows = data.gallery.filter((row) => $('case-overlap').value === 'all' || row.target === Number($('case-overlap').value));
  if (selectedField) rows = rows.filter((row) => row.source === selectedField);
  const sort = $('case-sort').value;
  if (sort === 'risk') rows.sort((a,b) => (b.occluder_false_weed_pixels / Math.max(1,b.occluder_crop_pixels)) - (a.occluder_false_weed_pixels / Math.max(1,a.occluder_crop_pixels)));
  if (sort === 'miss') rows.sort((a,b) => a.visible_weed_recall - b.visible_weed_recall);
  if (sort === 'source') rows.sort((a,b) => a.source.localeCompare(b.source) || a.target - b.target);
  return rows;
}

function showCase() {
  const list = cases();
  if (!list.length) return;
  caseIndex = ((caseIndex % list.length) + list.length) % list.length;
  const row = list[caseIndex];
  $('case-counter').textContent = `${String(caseIndex+1).padStart(2,'0')} / ${String(list.length).padStart(2,'0')}${selectedField ? ` · IMAGE ${selectedField}` : ''}`;
  $('case-id').textContent = row.id;
  $('case-coverage').textContent = pct(row.actual);
  $('case-recall').textContent = row.visible_pixels < 10 ? 'N/A' : pct(row.visible_weed_recall);
  $('case-risk').textContent = row.occluder_crop_pixels ? pct(row.occluder_false_weed_pixels / row.occluder_crop_pixels) : '—';
  $('source-image').src = row.images.rgb;
  $('analysis-image').src = row.images[imageView];
  $('analysis-caption').textContent = ({pred:'MODEL PREDICTION',gt:'GROUND TRUTH',heat:'WEED PROBABILITY'})[imageView];
}

function fieldAudit() {
  $('field-rows').innerHTML = data.per_image.map((row) => { const hasCases = data.gallery.some((item) => item.source === row.source); return `<tr data-source="${esc(row.source)}" class="${hasCases ? '' : 'no-cases'}" title="${hasCases ? `Open image ${esc(row.source)} cases` : 'No sampled occlusion case for this image'}"><td><b>${esc(row.source)}</b> ${hasCases ? '↗' : ''}</td><td>${row.weed_pixels ? pct(row.weed_iou) : '—'}</td><td>${row.crop_pixels ? pct(row.crop_iou) : '—'}</td><td class="${row.crop_false_weed_rate > .01 ? 'bad' : 'good'}">${row.crop_pixels ? pct(row.crop_false_weed_rate) : '—'}</td><td>${row.weed_pixels ? pct(row.weed_pixel_recall) : '—'}</td></tr>`; }).join('');
  $('field-rows').querySelectorAll('tr:not(.no-cases)').forEach((tr) => tr.addEventListener('click', () => {
    selectedField = tr.dataset.source; caseIndex = 0; showCase(); $('cases').scrollIntoView({behavior:'smooth'});
    $('clear-field').hidden = false;
  }));
  $('brier').textContent = data.clean.brier_plant.toFixed(3);
  $('calibration-bars').innerHTML = data.calibration.filter((row) => row.count > 0).map((row) => `<div class="cal-row" title="${row.count.toLocaleString()} plant pixels with predicted weed probability ${Math.round(row.bin*100)}–${Math.round((row.bin+.1)*100)}%"><span>${Math.round(row.bin*100)}–${Math.round((row.bin+.1)*100)}%</span><div><i style="width:${row.accuracy*100}%"></i><b style="left:${row.confidence*100}%"></b></div><em>${pct(row.accuracy,0)}</em></div>`).join('');
}

async function start() {
  try {
    const response = await fetch('results.json');
    const json = await response.json();
    if (!response.ok || !json.clean || !json.sweep) throw new Error('Run the segmentation training and evaluation commands in README.md');
    data = json;
    $('loading').hidden = true; $('content').hidden = false;
    $('verdict-title').textContent = data.illustrative_gate.passed ? 'PASS · candidate for further validation' : 'NO-GO · crop-risk gate failed';
    $('verdict').classList.toggle('pass', data.illustrative_gate.passed);
    const missed = [data.clean.crop_false_weed_rate > data.illustrative_gate.max_crop_false_weed_rate ? 'crop error' : null, data.clean.weed_pixel_recall < data.illustrative_gate.min_weed_pixel_recall ? 'weed recall' : null].filter(Boolean);
    $('verdict-text').textContent = `${pct(data.clean.crop_false_weed_rate)} of labeled crop pixels were called weed. Weed recall was ${pct(data.clean.weed_pixel_recall)}. ${missed.length ? `This misses the ${missed.join(' and ')} criterion.` : 'Both example criteria pass on this held-out split.'}`;
    $('weed-iou').textContent = data.clean.iou_weed.toFixed(3);
    $('crop-risk').textContent = pct(data.clean.crop_false_weed_rate);
    $('crop-risk-ci').textContent = `95% image-bootstrap interval ${pct(data.bootstrap_95.crop_false_weed_rate[0],0)}–${pct(data.bootstrap_95.crop_false_weed_rate[1],0)}`;
    $('weed-recall').textContent = pct(data.clean.weed_pixel_recall);
    $('latency').textContent = `${data.clean.latency_ms_per_image.toFixed(1)} ms`;
    thresholdIndex = data.operating_curve.reduce((best, row, i) => Math.abs(row.threshold-data.threshold) < Math.abs(data.operating_curve[best].threshold-data.threshold) ? i : best, 0);
    $('threshold').max = String(data.operating_curve.length-1);
    $('threshold').value = String(thresholdIndex);
    $('stress-count').textContent = `${data.stress_sample_count} WEED COMPONENTS`;
    
    operatingChart(); overlapChart(); fieldAudit(); showCase();
    $('hero-image').src = data.gallery.find((item) => item.target === .4)?.images.rgb || data.gallery[0].images.rgb;
    $('threshold').addEventListener('input', (event) => { thresholdIndex = Number(event.target.value); operatingChart(); });
    $('case-overlap').addEventListener('change', () => { caseIndex = 0; showCase(); });
    $('case-sort').addEventListener('change', () => { caseIndex = 0; showCase(); });
    $('prev').addEventListener('click', () => { caseIndex--; showCase(); });
    $('next').addEventListener('click', () => { caseIndex++; showCase(); });
    $('clear-field').addEventListener('click', () => { selectedField = null; caseIndex = 0; $('clear-field').hidden = true; showCase(); });
    document.querySelectorAll('[data-view]').forEach((button) => button.addEventListener('click', () => { imageView = button.dataset.view; document.querySelectorAll('[data-view]').forEach((x) => x.classList.toggle('selected', x === button)); showCase(); }));
    $('provenance').textContent = `Prepared mask fingerprint ${data.source_fingerprint}. Epoch ${data.checkpoint_epoch}. ${data.split.train.images}/${data.split.val.images}/${data.split.test.images} train/validation/test tiles, grouped in eights by original photo. Weed threshold ${data.threshold.toFixed(3)}.`;
  } catch (error) { $('content').hidden = true; $('loading').hidden = false; $('loading').textContent = `${error.message}.`; }
}
start();
