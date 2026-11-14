'use strict';
// Read-only pull of Item Ledger Entries for one brand (Global Dimension 2) per month -> NDJSON, resumable.
// usage: node extract/ile-pull.js <BRAND> <yyyy-mm from> <yyyy-mm to> <COMPANY_ID> [outdir]
// credentials: BC_TENANT_ID, BC_CLIENT_ID, BC_CLIENT_SECRET, BC_SCOPE from the environment or from the file named by ENV_FILE (default ./.env)
// BC_TARGET_ENV selects the ERP environment (default Production); it is deliberately NOT read from a shared .env.
const fs = require('fs'), path = require('path');
const envFile = process.env.ENV_FILE || '.env';
if (fs.existsSync(envFile)) for (const l of fs.readFileSync(envFile, 'utf8').split('\n')) {
  const m = l.match(/^([A-Z0-9_]+)=(.*)$/); if (m && !process.env[m[1]]) process.env[m[1]] = m[2].trim();
}
const { BC_TENANT_ID: T, BC_CLIENT_ID: C, BC_CLIENT_SECRET: S, BC_SCOPE: SC } = process.env;
const cfgFile = process.env.BC_FIELDS_FILE || 'bc_fields.local.json';
const SERVICE = (fs.existsSync(cfgFile) ? JSON.parse(fs.readFileSync(cfgFile, 'utf8')).ledger_service : null) || 'Item_Ledger_Entry';
const TARGET_ENV = process.env.BC_TARGET_ENV || 'Production';
const [brand, from, to, company, outdir = 'workdir/raw'] = process.argv.slice(2);
if (!brand || !from || !to || !company) { console.error('usage: node extract/ile-pull.js BRAND yyyy-mm yyyy-mm COMPANY_ID [outdir]'); process.exit(1); }
const BASE = `https://api.businesscentral.dynamics.com/v2.0/${T}/${TARGET_ENV}/ODataV4/Company('${company}')/${SERVICE}`;
const FIELDS = ['Entry_No','Posting_Date','Entry_Type','Document_Type','Document_No','Item_No','Location_Code','Global_Dimension_1_Code','Global_Dimension_2_Code','Quantity','Sales_Amount_Actual','Cost_Amount_Actual','Order_No'].join(',');
let tok, tokAt = 0;
async function token() {
  if (tok && Date.now() - tokAt < 40 * 60e3) return tok;
  const r = await fetch(`https://login.microsoftonline.com/${T}/oauth2/v2.0/token`, { method: 'POST', headers: { 'Content-Type': 'application/x-www-form-urlencoded' }, body: new URLSearchParams({ grant_type: 'client_credentials', client_id: C, client_secret: S, scope: SC }) });
  tok = (await r.json()).access_token; tokAt = Date.now(); return tok;
}
async function get(url) {
  for (let i = 0; i < 6; i++) {
    try {
      const x = await fetch(url, { headers: { Authorization: 'Bearer ' + await token(), Accept: 'application/json' } });
      if (x.ok) return await x.json();
      console.error('HTTP', x.status, '- retry', i + 1);
    } catch (e) { console.error(e.message, '- retry', i + 1); }
    await new Promise(r => setTimeout(r, 3000 * (i + 1)));
  }
  throw new Error('giving up: ' + url.slice(0, 150));
}
function months(a, b) { const out = []; let [y, m] = a.split('-').map(Number); const [y2, m2] = b.split('-').map(Number);
  while (y < y2 || (y === y2 && m <= m2)) { out.push([y, m]); m++; if (m > 12) { m = 1; y++; } } return out; }
(async () => {
  fs.mkdirSync(outdir, { recursive: true });
  for (const [y, m] of months(from, to)) {
    const tag = `${brand}_${company}_${y}-${String(m).padStart(2, '0')}`;
    const f = path.join(outdir, tag + '.ndjson'), done = f + '.done';
    if (fs.existsSync(done)) { console.log(tag, 'skip (done)'); continue; }
    const ny = m === 12 ? y + 1 : y, nm = m === 12 ? 1 : m + 1;
    const d1 = `${y}-${String(m).padStart(2, '0')}-01`, d2 = `${ny}-${String(nm).padStart(2, '0')}-01`;
    const flt = encodeURIComponent(`Global_Dimension_2_Code eq '${brand}' and Posting_Date ge ${d1} and Posting_Date lt ${d2}`);
    let url = `${BASE}?$filter=${flt}&$select=${FIELDS}`, n = 0;
    const ws = fs.createWriteStream(f);
    while (url) { const j = await get(url); for (const e of j.value) { delete e['@odata.etag']; ws.write(JSON.stringify(e) + '\n'); n++; } url = j['@odata.nextLink'] || null; }
    await new Promise(r => ws.end(r)); fs.writeFileSync(done, String(n));
    console.log(tag, n, 'rows');
  }
})().catch(e => { console.error(e); process.exit(1); });
