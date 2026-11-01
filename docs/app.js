/*
 * Vajra Image Guard - UI layer. Everything here is presentation, canvas work and
 * file I/O; the protection maths lives in engine.js.
 */
(function () {
  'use strict';

  var MAX_PIXELS = 24e6;
  var $ = function (id) { return document.getElementById(id); };

  var state = {
    items: [],
    active: -1,
    view: 'original',
    result: null,
    gain: 8
  };

  /* ------------------------------------------------------------- elements -- */

  var el = {
    drop: $('drop'), file: $('file'), queue: $('queue'),
    strength: $('strength'), strengthVal: $('strengthVal'),
    iters: $('iters'), itersVal: $('itersVal'),
    seed: $('seed'), seedBtn: $('seedBtn'),
    objective: $('objective'),
    maskFloor: $('maskFloor'), maskFloorVal: $('maskFloorVal'),
    jitter: $('jitter'), jitterVal: $('jitterVal'),
    run: $('run'), status: $('status'),
    stage: $('stage'), empty: $('empty'), busy: $('busy'), busyText: $('busyText'),
    metrics: $('metrics'),
    mPsnr: $('mPsnr'), mSsim: $('mSsim'), mMax: $('mMax'), mMean: $('mMean'),
    mSupp: $('mSupp'), mRes: $('mRes'), mTime: $('mTime'),
    exportRow: $('export'),
    dlPng: $('dlPng'), dlJpg: $('dlJpg'), quality: $('quality'), share: $('share'), saveNote: $('saveNote'),
    probe: $('probe'), probeQ: $('probeQ'), probeRun: $('probeRun'), probeOut: $('probeOut'),
    gainWrap: $('gainWrap'), gain: $('gain'), gainVal: $('gainVal'),
    tabs: Array.prototype.slice.call(document.querySelectorAll('.tab'))
  };

  var ctx = el.stage.getContext('2d');
  var scratch = document.createElement('canvas');
  var scratchCtx = scratch.getContext('2d');

  /* ---------------------------------------------------------------- utils -- */

  function setStatus(msg, isError) {
    el.status.textContent = msg;
    el.status.classList.toggle('is-error', !!isError);
  }

  function busy(on, text) {
    el.busy.hidden = !on;
    if (text) el.busyText.textContent = text;
  }

  // Yield to the browser so the spinner actually paints before the solve.
  function nextFrame() {
    return new Promise(function (res) {
      requestAnimationFrame(function () { setTimeout(res, 0); });
    });
  }

  function loadImage(file) {
    return new Promise(function (resolve, reject) {
      var url = URL.createObjectURL(file);
      var img = new Image();
      img.onload = function () { URL.revokeObjectURL(url); resolve(img); };
      img.onerror = function () { URL.revokeObjectURL(url); reject(new Error('Could not decode ' + file.name)); };
      img.src = url;
    });
  }

  function targetSize(w, h) {
    var scale = Math.min(1, Math.sqrt(MAX_PIXELS / (w * h)));
    return { w: Math.max(1, Math.round(w * scale)), h: Math.max(1, Math.round(h * scale)) };
  }

  function imageToData(img) {
    var t = targetSize(img.naturalWidth || img.width, img.naturalHeight || img.height);
    scratch.width = t.w;
    scratch.height = t.h;
    scratchCtx.clearRect(0, 0, t.w, t.h);
    scratchCtx.drawImage(img, 0, 0, t.w, t.h);
    return scratchCtx.getImageData(0, 0, t.w, t.h);
  }

  // putImageData only accepts a real ImageData, not an object with the same
  // shape, so wrap whatever the engine hands back.
  function asImageData(obj) {
    if (typeof ImageData === 'function' && obj instanceof ImageData) return obj;
    var src = obj.data;
    var buf = src instanceof Uint8ClampedArray ? src : new Uint8ClampedArray(src);
    return new ImageData(buf, obj.width, obj.height);
  }

  function dataToCanvas(canvas, ctx2, imageData) {
    canvas.width = imageData.width;
    canvas.height = imageData.height;
    ctx2.putImageData(asImageData(imageData), 0, 0);
  }

  // Optional hardening: rotate by a fraction of a degree and rescale, so the
  // protected pixels no longer sit on the original grid.
  function microJitter(imageData, degrees) {
    if (!degrees) return imageData;
    var w = imageData.width;
    var h = imageData.height;
    var tmp = document.createElement('canvas');
    tmp.width = w;
    tmp.height = h;
    var tctx = tmp.getContext('2d');
    tctx.putImageData(imageData, 0, 0);

    var out = document.createElement('canvas');
    out.width = w;
    out.height = h;
    var octx = out.getContext('2d');
    octx.imageSmoothingEnabled = true;
    octx.imageSmoothingQuality = 'high';
    octx.translate(w / 2, h / 2);
    octx.rotate((degrees * Math.PI) / 180);
    var zoom = 1 + degrees * 0.002;
    octx.scale(zoom, zoom);
    octx.drawImage(tmp, -w / 2, -h / 2);
    return octx.getImageData(0, 0, w, h);
  }

  function amplifiedDiff(a, b, gain) {
    var out = new Uint8ClampedArray(a.data.length);
    for (var i = 0; i < a.data.length; i += 4) {
      out[i] = 128 + (b.data[i] - a.data[i]) * gain;
      out[i + 1] = 128 + (b.data[i + 1] - a.data[i + 1]) * gain;
      out[i + 2] = 128 + (b.data[i + 2] - a.data[i + 2]) * gain;
      out[i + 3] = 255;
    }
    return { data: out, width: a.width, height: a.height };
  }

  function jpegRoundTrip(imageData, quality) {
    var w = imageData.width;
    var h = imageData.height;
    var c = document.createElement('canvas');
    c.width = w;
    c.height = h;
    c.getContext('2d').putImageData(asImageData(imageData), 0, 0);

    function decode(source) {
      var c2 = document.createElement('canvas');
      c2.width = w;
      c2.height = h;
      var x2 = c2.getContext('2d');
      x2.drawImage(source, 0, 0);
      return x2.getImageData(0, 0, w, h);
    }

    return new Promise(function (resolve, reject) {
      c.toBlob(function (blob) {
        if (!blob) { reject(new Error('JPEG encode failed')); return; }
        if (typeof createImageBitmap === 'function') {
          createImageBitmap(blob).then(function (bmp) {
            var data = decode(bmp);
            if (bmp.close) bmp.close();
            resolve(data);
          }).catch(function () { reject(new Error('JPEG decode failed')); });
          return;
        }
        var url = URL.createObjectURL(blob);
        var img = new Image();
        img.onload = function () { URL.revokeObjectURL(url); resolve(decode(img)); };
        img.onerror = function () { URL.revokeObjectURL(url); reject(new Error('JPEG decode failed')); };
        img.src = url;
      }, 'image/jpeg', Math.min(1, Math.max(0.05, quality)));
    });
  }

  function luma(data, i) {
    return 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
  }

  // Pearson correlation between the applied perturbation and what is left of it
  // after a JPEG round trip: the number a user actually cares about.
  function survival(original, protectedData, reOriginal, reProtected) {
    var n = original.data.length / 4;
    var sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
    for (var p = 0; p < n; p++) {
      var i = p * 4;
      var applied = luma(protectedData.data, i) - luma(original.data, i);
      var kept = luma(reProtected.data, i) - luma(reOriginal.data, i);
      sx += applied; sy += kept;
      sxx += applied * applied; syy += kept * kept; sxy += applied * kept;
    }
    var cov = sxy / n - (sx / n) * (sy / n);
    var vx = sxx / n - (sx / n) * (sx / n);
    var vy = syy / n - (sy / n) * (sy / n);
    if (vx <= 1e-9 || vy <= 1e-9) return { corr: 0, retained: 0 };
    var corr = cov / Math.sqrt(vx * vy);
    var retained = Math.sqrt(vy / vx);
    return { corr: corr, retained: retained };
  }

  /* ------------------------------------------------------------ queue/load -- */

  function addFiles(files) {
    var list = Array.prototype.slice.call(files).filter(function (f) {
      return f && /^image\//.test(f.type);
    });
    if (!list.length) {
      setStatus('That file is not an image.', true);
      return;
    }
    Promise.all(list.map(function (f) {
      return loadImage(f).then(function (img) { return { name: f.name, img: img, result: null }; });
    })).then(function (items) {
      state.items = state.items.concat(items);
      renderQueue();
      selectItem(state.items.length - items.length);
      setStatus(items.length > 1
        ? items.length + ' images queued. Protect applies to the selected one.'
        : 'Ready. Press Protect.');
    }).catch(function (err) {
      setStatus(err.message, true);
    });
  }

  function renderQueue() {
    el.queue.innerHTML = '';
    el.queue.hidden = state.items.length === 0;
    state.items.forEach(function (item, idx) {
      var b = document.createElement('button');
      b.className = 'queue-item' + (idx === state.active ? ' is-active' : '') + (item.result ? ' is-done' : '');
      b.type = 'button';
      b.title = item.name;
      var thumb = document.createElement('img');
      thumb.alt = item.name;
      thumb.src = item.img.src;
      b.appendChild(thumb);
      b.addEventListener('click', function () { selectItem(idx); });
      el.queue.appendChild(b);
    });
  }

  function selectItem(idx) {
    state.active = idx;
    var item = state.items[idx];
    if (!item) return;
    state.result = item.result;
    el.run.disabled = false;
    renderQueue();
    if (item.result) {
      showResult(item.result);
    } else {
      var data = imageToData(item.img);
      dataToCanvas(el.stage, ctx, data);
      el.stage.classList.add('is-visible');
      el.empty.hidden = true;
      el.metrics.hidden = true;
      el.exportRow.hidden = true;
      el.probe.hidden = true;
      setView('original');
    }
  }

  /* --------------------------------------------------------------- process -- */

  function options() {
    return {
      epsilon: Number(el.strength.value) / 255,
      iterations: Number(el.iters.value),
      seed: Number(el.seed.value) || 0,
      objective: el.objective.value,
      maskFloor: Number(el.maskFloor.value) / 100,
      workingSide: 512,
      maxPixels: MAX_PIXELS
    };
  }

  function protect() {
    var item = state.items[state.active];
    if (!item) return;
    var opts = options();
    var jitter = Number(el.jitter.value) / 10; // 0 .. 1 degree

    el.run.disabled = true;
    setStatus('Solving…');
    busy(true, 'Protecting…');

    nextFrame().then(function () {
      var started = performance.now();
      var source = imageToData(item.img);
      var res = Vajra.protect(source, opts);
      var protectedData = { data: res.data, width: res.width, height: res.height };
      if (jitter > 0) protectedData = microJitter(protectedData, jitter);
      var elapsed = Math.round(performance.now() - started);

      item.result = {
        source: source,
        protectedData: protectedData,
        stats: res.stats,
        elapsed: elapsed,
        opts: opts
      };
      state.result = item.result;
      renderQueue();
      showResult(item.result);
      setStatus('Protected in ' + elapsed + ' ms. Nothing left your device.');
      el.run.disabled = false;
      busy(false);
    }).catch(function (err) {
      busy(false);
      el.run.disabled = false;
      setStatus(err.message || 'Protection failed.', true);
    });
  }

  function showResult(result) {
    var s = result.stats;
    el.mPsnr.textContent = isFinite(s.psnr) ? s.psnr.toFixed(2) + ' dB' : '∞';
    el.mSsim.textContent = s.ssim.toFixed(4);
    el.mMax.textContent = s.maxDelta.toFixed(0) + ' / 255';
    el.mMean.textContent = s.meanDelta.toFixed(2) + ' / 255';
    el.mSupp.textContent = Math.round(s.suppression * 100) + '%';
    el.mRes.textContent = s.workingWidth + '×' + s.workingHeight;
    el.mTime.textContent = result.elapsed + ' ms';
    el.metrics.hidden = false;
    el.exportRow.hidden = false;
    el.probe.hidden = false;
    el.empty.hidden = true;
    el.stage.classList.add('is-visible');
    el.probeOut.textContent = '';
    el.saveNote.textContent = '';
    enableResultTabs(true);
    setView(state.view === 'original' ? 'protected' : state.view);
  }

  function enableResultTabs(on) {
    el.tabs.forEach(function (t) {
      if (t.dataset.view !== 'original') t.disabled = !on;
    });
  }

  function setView(view) {
    state.view = view;
    var item = state.items[state.active];
    if (!item) return;
    el.tabs.forEach(function (t) { t.classList.toggle('is-active', t.dataset.view === view); });
    el.gainWrap.hidden = view !== 'diff';

    if (!item.result) {
      dataToCanvas(el.stage, ctx, imageToData(item.img));
      return;
    }
    var r = item.result;
    if (view === 'original') dataToCanvas(el.stage, ctx, r.source);
    else if (view === 'protected') dataToCanvas(el.stage, ctx, r.protectedData);
    else dataToCanvas(el.stage, ctx, amplifiedDiff(r.source, r.protectedData, state.gain));
  }

  /* ---------------------------------------------------------------- export -- */

  function currentName(suffix, ext) {
    var item = state.items[state.active];
    var base = (item && item.name ? item.name : 'image').replace(/\.[^.]+$/, '');
    return base + suffix + '.' + ext;
  }

  function download(dataUrl, name) {
    var a = document.createElement('a');
    a.href = dataUrl;
    a.download = name;
    document.body.appendChild(a);
    a.click();
    a.remove();
  }

  function exportAs(type) {
    var item = state.items[state.active];
    if (!item || !item.result) return;
    var r = item.result;
    var c = document.createElement('canvas');
    c.width = r.protectedData.width;
    c.height = r.protectedData.height;
    c.getContext('2d').putImageData(asImageData(r.protectedData), 0, 0);

    if (type === 'png') {
      var png = c.toDataURL('image/png');
      deliver(png, currentName('-vajra', 'png'));
    } else {
      var q = Math.min(1, Math.max(0.4, Number(el.quality.value) / 100));
      var jpg = c.toDataURL('image/jpeg', q);
      deliver(jpg, currentName('-vajra', 'jpg'));
    }
  }

  // On Android the shell exposes a native bridge; everywhere else fall back to
  // a normal browser download.
  function deliver(dataUrl, name) {
    var bridge = window.VajraAndroid;
    if (bridge && typeof bridge.saveImage === 'function') {
      try {
        var ok = bridge.saveImage(dataUrl, name);
        el.saveNote.textContent = ok ? 'Saved to Pictures/Vajra' : 'Save failed';
        return;
      } catch (e) {
        el.saveNote.textContent = 'Save failed';
        return;
      }
    }
    download(dataUrl, name);
    el.saveNote.textContent = 'Downloaded ' + name;
  }

  function share() {
    var item = state.items[state.active];
    if (!item || !item.result) return;
    var r = item.result;
    var c = document.createElement('canvas');
    c.width = r.protectedData.width;
    c.height = r.protectedData.height;
    c.getContext('2d').putImageData(asImageData(r.protectedData), 0, 0);

    // Inside the Android shell, hand the PNG to the native share sheet.
    var bridge = window.VajraAndroid;
    if (bridge && typeof bridge.shareImage === 'function') {
      var ok = false;
      try {
        ok = bridge.shareImage(c.toDataURL('image/png'), currentName('-vajra', 'png'));
      } catch (e) {
        ok = false;
      }
      el.saveNote.textContent = ok ? 'Saved to Pictures/Vajra and opened the share sheet' : 'Share failed';
      return;
    }

    c.toBlob(function (blob) {
      if (!blob) return;
      var file = new File([blob], currentName('-vajra', 'png'), { type: 'image/png' });
      if (navigator.canShare && navigator.canShare({ files: [file] })) {
        navigator.share({ files: [file], title: 'Protected image' }).catch(function () {});
      } else {
        el.saveNote.textContent = 'Share is not available in this browser — download instead.';
      }
    }, 'image/png');
  }

  /* ----------------------------------------------------------------- probe -- */

  function runProbe() {
    var item = state.items[state.active];
    if (!item || !item.result) return;
    var q = Math.min(1, Math.max(0.3, Number(el.probeQ.value) / 100));
    el.probeRun.disabled = true;
    el.probeOut.textContent = 'measuring…';

    Promise.all([
      jpegRoundTrip(item.result.source, q),
      jpegRoundTrip(item.result.protectedData, q)
    ]).then(function (out) {
      var s = survival(item.result.source, item.result.protectedData, out[0], out[1]);
      var pct = Math.max(0, s.corr) * 100;
      var verdict = pct > 65 ? 'holds up' : pct > 35 ? 'partly survives' : 'mostly lost';
      el.probeOut.textContent = pct.toFixed(0) + '% of the perturbation survives JPEG q' +
        Math.round(q * 100) + ' — ' + verdict;
      el.probeRun.disabled = false;
    }).catch(function (err) {
      el.probeOut.textContent = 'probe error: ' + (err.message || String(err));
      el.probeRun.disabled = false;
    });
  }

  /* ----------------------------------------------------------------- wiring -- */

  el.drop.addEventListener('click', function () { el.file.click(); });
  el.drop.addEventListener('keydown', function (e) {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); el.file.click(); }
  });
  el.file.addEventListener('change', function () {
    if (el.file.files && el.file.files.length) addFiles(el.file.files);
    el.file.value = '';
  });

  ['dragenter', 'dragover'].forEach(function (ev) {
    el.drop.addEventListener(ev, function (e) { e.preventDefault(); el.drop.classList.add('is-over'); });
  });
  ['dragleave', 'drop'].forEach(function (ev) {
    el.drop.addEventListener(ev, function (e) { e.preventDefault(); el.drop.classList.remove('is-over'); });
  });
  el.drop.addEventListener('drop', function (e) {
    if (e.dataTransfer && e.dataTransfer.files) addFiles(e.dataTransfer.files);
  });
  // Dropping anywhere on the stage works too.
  document.addEventListener('dragover', function (e) { e.preventDefault(); });
  document.addEventListener('drop', function (e) {
    e.preventDefault();
    if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length) addFiles(e.dataTransfer.files);
  });

  el.strength.addEventListener('input', function () {
    el.strengthVal.textContent = el.strength.value + ' / 255';
  });
  el.iters.addEventListener('input', function () { el.itersVal.textContent = el.iters.value; });
  el.maskFloor.addEventListener('input', function () {
    el.maskFloorVal.textContent = el.maskFloor.value + '%';
  });
  el.jitter.addEventListener('input', function () {
    var v = Number(el.jitter.value);
    el.jitterVal.textContent = v === 0 ? 'off' : (v / 10).toFixed(1) + '°';
  });
  el.gain.addEventListener('input', function () {
    state.gain = Number(el.gain.value);
    el.gainVal.textContent = state.gain + '×';
    if (state.view === 'diff') setView('diff');
  });
  el.seedBtn.addEventListener('click', function () {
    el.seed.value = String(Math.floor(Math.random() * 1e6));
  });

  el.run.addEventListener('click', protect);
  el.dlPng.addEventListener('click', function () { exportAs('png'); });
  el.dlJpg.addEventListener('click', function () { exportAs('jpeg'); });
  el.share.addEventListener('click', share);
  el.probeRun.addEventListener('click', runProbe);

  el.tabs.forEach(function (t) {
    t.addEventListener('click', function () {
      if (!t.disabled) setView(t.dataset.view);
    });
  });

  if (navigator.canShare || (window.VajraAndroid && window.VajraAndroid.shareImage)) el.share.hidden = false;

  /* ------------------------------------------------------------------- pwa -- */

  if ('serviceWorker' in navigator && location.protocol !== 'file:') {
    window.addEventListener('load', function () {
      navigator.serviceWorker.register('sw.js').catch(function () { /* offline caching is optional */ });
    });
  }

  var installEvent = null;
  window.addEventListener('beforeinstallprompt', function (e) {
    e.preventDefault();
    installEvent = e;
    var btn = $('installBtn');
    if (btn) btn.hidden = false;
  });
  var installBtn = $('installBtn');
  if (installBtn) {
    installBtn.addEventListener('click', function () {
      if (!installEvent) return;
      installEvent.prompt();
      installEvent = null;
      installBtn.hidden = true;
    });
  }

  // Initial UI state
  el.strengthVal.textContent = el.strength.value + ' / 255';
  el.itersVal.textContent = el.iters.value;
  el.maskFloorVal.textContent = el.maskFloor.value + '%';
  if (window.VajraAndroid) el.saveNote.textContent = '';
})();
