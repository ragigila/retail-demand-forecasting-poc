'use strict';
// Read-only pull of Item Card attributes ONLY for items that appear in the extracted ledger files; batches of N items per request.
// usage: node extract/item-pull.js <COMPANY_ID> [rawdir] [outfile]
// needs bc_fields.local.json (see bc_fields.example.json) with item_service and item_fields; credentials as in ile-pull.js
const fs = require('fs'), path = require('path');
const envFile = process.env.ENV_FILE || '.env';
if (fs.existsSync(envFile)) for (const l of fs.readFileSync(envFile, 'utf8').split('\n')) {
  const m = l.match(/^([A-Z0-9_]+)=(.*)$/); if (m && !process.env[m[1]]) process.env[m[1]] = m[2].trim();
}
const { BC_TENANT_ID: T, BC_CLIENT_ID: C, BC_CLIENT_SECRET: S, BC_SCOPE: SC } = process.env;
const cfgFile = process.env.BC_FIELDS_FILE || 'bc_fields.local.json';
if (!fs.existsSync(cfgFile)) { console.error('missing ' + cfgFile + ' (copy bc_fields.example.json)'); process.exit(1); }
const CFG = JSON.parse(fs.readFileSync(cfgFile, 'utf8'));
const TARGET_ENV = process.env.BC_TARGET_ENV || 'Production';
const [company, rawdir = 'workdir/raw', out = 'workdir/raw/items.ndjson'] = process.argv.slice(2);
if (!company) { console.error('usage: node extract/item-pull.js COMPANY_ID [rawdir] [outfile]'); process.exit(1); }
const BASE = `https://api.businesscentral.dynamics.com/v2.0/${T}/${TARGET_ENV}/ODataV4/Company('${company}')/${CFG.item_service}`;
const FIELDS = CFG.item_fields.join(',');
const N = 40, CONC = 4;
let tok, tokAt = 0;
async function token() {
  if (tok && Date.now() - tokAt < 40 * 60e3) return tok;
  const r = await fetch(`https://login.microsoftonline.com/${T}/oauth2/v2.0/token`, { method: 'POST', headers: { 'Content-Type': 'application/x-www-form-urlencoded' }, body: new URLSearchParams({ grant_type: 'client_credentials', client_id: C, client_secret: S, scope: SC }) });
  tok = (await r.json()).access_token; tokAt = Date.now(); return tok;
}
async function get(url) {
  for (let i = 0; i < 6; i++) {
    try { const x = await fetch(url, { headers: { Authorization: 'Bearer ' + await token() } }); if (x.ok) return await x.json(); console.error('HTTP', x.status); }
    catch (e) { console.error(e.message); }
    await new Promise(r => setTimeout(r, 3000 * (i + 1)));
  }
  throw new Error('giving up');
}
(async () => {
  const ids = new Set();
  for (const f of fs.readdirSync(rawdir).filter(f => f.endsWith('.ndjson.done'))) {
    const nd = path.join(rawdir, f.replace('.done', ''));
    for (const line of fs.readFileSync(nd, 'utf8').split('\n')) { if (!line) continue; ids.add(JSON.parse(line).Item_No); }
  }
  const list = [...ids]; console.log('distinct items in ledger', list.length);
  const batches = []; for (let i = 0; i < list.length; i += N) batches.push(list.slice(i, i + N));
  const ws = fs.createWriteStream(out); let done = 0, rows = 0, next = 0;
  async function worker() {
    while (next < batches.length) {
      const b = batches[next++];
      const flt = b.map(x => `No eq '${x.replace(/'/g, "''")}'`).join(' or ');
      let url = `${BASE}?$filter=${encodeURIComponent(flt)}&$select=${FIELDS}`;
      while (url) { const j = await get(url); for (const e of j.value) { delete e['@odata.etag']; ws.write(JSON.stringify(e) + '\n'); rows++; } url = j['@odata.nextLink'] || null; }
      if (++done % 50 === 0) console.log(done, '/', batches.length, 'batches', rows, 'rows');
    }
  }
  await Promise.all(Array.from({ length: CONC }, worker));
  await new Promise(r => ws.end(r)); console.log('items written', rows);
})().catch(e => { console.error(e); process.exit(1); });
