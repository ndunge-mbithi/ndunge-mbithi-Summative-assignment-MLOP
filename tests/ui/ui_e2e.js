// End-to-end UI check: drives the real web UI against a running stack and saves screenshots.
//   npm install @sparticuz/chromium@131 puppeteer-core@23   (or set CHROME_PATH to any Chrome/Chromium)
//   BASE_URL=http://localhost:8080 node tests/ui/ui_e2e.js
const path = require('path');
const puppeteer = require('puppeteer-core');
const BASE = process.env.BASE_URL || 'http://localhost:8080';
const OUT = path.resolve(__dirname, '../../docs/screenshots');
const ZIP = path.resolve(__dirname, '../../demo_upload/demo_batch.zip');
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

(async () => {
  let exe = process.env.CHROME_PATH, args = [];
  if (!exe) { const c = require('@sparticuz/chromium'); exe = await c.executablePath(); args = c.args; }
  const browser = await puppeteer.launch({ executablePath: exe, args: [...args, '--no-proxy-server'], headless: true,
    defaultViewport: { width: 1360, height: 900, deviceScaleFactor: 1 } });
  const page = await browser.newPage();
  const errors = [];
  page.on('pageerror', (e) => errors.push(e.message));
  page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
  const tab = async (name) => { await page.click(`.tabs button[data-tab="${name}"]`); await sleep(1500); };
  const shot = async (name) => { await page.screenshot({ path: `${OUT}/${name}.png`, fullPage: true }); console.log('screenshot', name); };

  await page.goto(BASE + '/', { waitUntil: 'networkidle0' });
  await page.waitForFunction(() => document.querySelector('#st-model').textContent.startsWith('model v'));

  // 1. single prediction on a known parasitized and a known uninfected test cell
  await page.click('details.hint summary');
  for (const kind of ['uninfected', 'parasitized']) {
    await page.click(`[data-example="${kind}"]`);
    await page.waitForFunction(() => !document.querySelector('#pred-btn').disabled);
    await page.click('#pred-btn');
    await page.waitForFunction(() => document.querySelector('#pred-state').classList.contains('ok'), { timeout: 30000 });
    const verdict = await page.$eval('#pred-label', (e) => e.textContent);
    const file = await page.$eval('#pred-drop-text', (e) => e.textContent);
    console.log(`predict ${kind}: ${verdict} (${file})`);
    if (verdict.toLowerCase() !== kind) throw new Error(`wrong prediction for ${kind}: ${verdict}`);
  }
  await page.click('[data-fb="Parasitized"]');
  await page.waitForFunction(() => document.querySelector('#fb-state').textContent.startsWith('recorded'));
  await shot('01_predict');

  // 2. dataset insights
  await tab('insights'); await sleep(1000); await shot('02_insights');

  // 3. bulk upload of a labelled ZIP, then retraining from the UI button
  await tab('data');
  const input = await page.$('#up-zip'); await input.uploadFile(ZIP);
  await page.type('#up-note', 'UI e2e - validation-split patients');
  await page.click('#up-btn');
  await page.waitForFunction(() => document.querySelector('#up-state').classList.contains('ok') || document.querySelector('#up-state').classList.contains('bad'), { timeout: 60000 });
  console.log('upload:', await page.$eval('#up-report', (e) => e.innerText.split('\n')[0]));
  await sleep(1500); await shot('03_upload');
  await page.click('#rt-btn');
  await page.waitForFunction(() => document.querySelector('#rt-state').classList.contains('ok') || document.querySelector('#rt-state').classList.contains('bad'), { timeout: 30000 });
  console.log('retrain:', await page.$eval('#rt-state', (e) => e.textContent));
  let midShot = false;
  for (let i = 0; i < 400; i++) {
    const st = await page.$eval('#job-status', (e) => e.textContent).catch(() => '');
    if (!midShot && (await page.$$('#job-stages li.now')).length) {
      const stage = await page.$eval('#job-stages li.now', (e) => e.textContent);
      if (stage === 'training') { await sleep(4000); await shot('04_retraining_in_progress'); midShot = true; }
    }
    if (['succeeded', 'rejected', 'failed'].includes(st)) { console.log('job finished:', st); break; }
    await sleep(2000);
  }
  await sleep(1500); await shot('05_retraining_result');
  console.log('result:', await page.$eval('#job-result', (e) => e.innerText.split('\n')[0]));

  // 4. models + monitoring
  await tab('models'); await shot('06_models');
  await tab('monitor'); await sleep(1000); await shot('07_monitoring');
  console.log('status bar:', await page.$eval('#st-model', (e) => e.textContent));
  if (errors.length) { console.log('browser errors:', errors); process.exitCode = 1; }
  await browser.close();
})().catch((e) => { console.error('UI E2E FAILED:', e.message); process.exit(1); });
