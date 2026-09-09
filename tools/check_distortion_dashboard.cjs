// Optional browser smoke test. No web server, no network, no package install.
const { chromium } = require('playwright');
const { pathToFileURL } = require('node:url');
const path = require('node:path');

(async () => {
  const file = process.argv[2];
  if (!file) throw new Error('usage: node check_distortion_dashboard.cjs <HTML> [screenshot]');
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const errors = [];
  try {
    const page = await browser.newPage();
    page.on('pageerror', err => errors.push(String(err)));
    await page.route('http**://**', route => route.abort());
    await page.goto(pathToFileURL(path.resolve(file)).href);
    for (const width of [1365, 390]) {
      await page.setViewportSize({ width, height: 900 });
      const overflow = await page.evaluate(() => document.documentElement.scrollWidth > innerWidth);
      if (overflow) throw new Error(`page overflow at ${width}`);
      await page.selectOption('#filter', 'distorted');
      const visible = await page.locator('tbody tr:visible').count();
      const total = await page.locator('tbody tr').count();
      if (!total || visible > total) throw new Error('invalid matrix/filter counts');
      const wrong = await page.locator('tbody tr:visible').evaluateAll(rows => rows.some(r => !r.dataset.tags.includes('distorted')));
      if (wrong) throw new Error('filter left unrelated rows visible');
      await page.selectOption('#filter', 'crash');
      await page.selectOption('#filter', 'all');
      if (await page.locator('tbody tr:visible').count() !== total) throw new Error('reset filter failed');
    }
    await page.emulateMedia({ colorScheme: 'dark' });
    await page.setViewportSize({ width: 1365, height: 900 });
    if (process.argv[3]) await page.screenshot({ path: process.argv[3] });
    if (errors.length) throw new Error(errors.join('\n'));
    console.log(JSON.stringify({ status: 'DASHBOARD_BROWSER_SMOKE_PASS', widths: [1365, 390], page_errors: errors, rows: await page.locator('tbody tr').count() }));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
