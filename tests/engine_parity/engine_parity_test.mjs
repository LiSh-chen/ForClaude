// Node正確性對拍：載入跟前端一樣的二進位1分K + data.json，用engine.js跑同樣的
// 參數組合，逐筆比對 parity_dump.py 產生的Python原始輸出。
//
// 前置條件：先執行 python scripts/export_strategy_lab.py 產生
// scripts/_strategy_lab_export/（二進位1分K + data.json），再執行
// python tests/engine_parity/parity_dump.py 產生 parity_out/，才能跑這支。
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";
const require = createRequire(import.meta.url);

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(HERE, "..", "..");
const CHART_DIR = path.join(REPO_ROOT, "scripts", "_strategy_lab_export");
const ENGINE_PATH = path.join(REPO_ROOT, "web", "strategy_lab", "engine.js");
const PARITY_DIR = path.join(HERE, "parity_out");
const Engine = require(ENGINE_PATH);

const records = JSON.parse(fs.readFileSync(path.join(CHART_DIR, "data.json"), "utf-8"));
const daily = Engine.dailyFromRecords(records);
console.log("daily rows:", daily.n, daily.dateStr[0], "~", daily.dateStr[daily.n - 1]);

const YEARS = [];
for (let y = 2001; y <= 2023; y++) YEARS.push(y);
const dayDaysArrays = [], nightDaysArrays = [];
for (const y of YEARS) {
  const dayBuf = fs.readFileSync(path.join(CHART_DIR, "years", `${y}_day.bin`));
  const nightBuf = fs.readFileSync(path.join(CHART_DIR, "years", `${y}_night.bin`));
  dayDaysArrays.push(Engine.parseMinuteBin(dayBuf.buffer.slice(dayBuf.byteOffset, dayBuf.byteOffset + dayBuf.byteLength)));
  nightDaysArrays.push(Engine.parseMinuteBin(nightBuf.buffer.slice(nightBuf.byteOffset, nightBuf.byteOffset + nightBuf.byteLength)));
}
const dayMinute = Engine.alignMinuteBlocks(daily, dayDaysArrays);
const nightMinute = Engine.alignMinuteBlocks(daily, nightDaysArrays);
console.log("day-session days with data:", dayMinute.filter(Boolean).length);
console.log("night-session days with data:", nightMinute.filter(Boolean).length);

let totalCases = 0, totalPass = 0;
const failures = [];

function cmpTrades(name, expected, actual, tol = 0.02) {
  totalCases++;
  const issues = [];
  if (expected.length !== actual.length) {
    issues.push(`count mismatch: python=${expected.length} js=${actual.length}`);
  }
  const n = Math.min(expected.length, actual.length);
  let firstDiffCount = 0;
  for (let i = 0; i < n; i++) {
    const e = expected[i], a = actual[i];
    const fields = ["direction", "entryDate", "entryTime", "exitDate", "exitTime"];
    for (const f of fields) {
      if (e[f] !== a[f]) {
        if (firstDiffCount < 5) issues.push(`row ${i} field ${f}: python=${e[f]} js=${a[f]}`);
        firstDiffCount++;
      }
    }
    for (const f of ["entryPrice", "exitPrice", "pnlPoints"]) {
      if (Math.abs(e[f] - a[f]) > tol) {
        if (firstDiffCount < 5) issues.push(`row ${i} field ${f}: python=${e[f]} js=${a[f]}`);
        firstDiffCount++;
      }
    }
    if (e.approxTime !== a.approxTime) {
      if (firstDiffCount < 5) issues.push(`row ${i} approxTime: python=${e.approxTime} js=${a.approxTime}`);
      firstDiffCount++;
    }
  }
  if (issues.length === 0) {
    totalPass++;
    console.log(`  OK   ${name}  (${actual.length} trades)`);
  } else {
    failures.push({ name, issues, expectedN: expected.length, actualN: actual.length });
    console.log(`  FAIL ${name}  python=${expected.length} js=${actual.length} trades, ${issues.length} field diffs (showing up to 5)`);
    for (const s of issues.slice(0, 5)) console.log(`       ${s}`);
  }
}

function loadExpected(name) {
  return JSON.parse(fs.readFileSync(path.join(PARITY_DIR, `${name}.json`), "utf-8"));
}

// ---- og ----
cmpTrades("og_default", loadExpected("og_default"),
  Engine.runOpeningRally(daily, dayMinute, { entryMoy: 525, exitMoy: 540 }));
cmpTrades("og_variant", loadExpected("og_variant"),
  Engine.runOpeningRally(daily, dayMinute, { entryMoy: 540, exitMoy: 555 }));

// ---- lu / re ----
const luReDefault = Engine.runLunchReversal(daily, dayMinute, { shortEntryMoy: 720, flipMoy: 750, longExitMoy: 780, maxLossPoints: null });
cmpTrades("lu_default", loadExpected("lu_default"), luReDefault.filter(t => t.leg === "short_lunch_dip"));
cmpTrades("re_default", loadExpected("re_default"), luReDefault.filter(t => t.leg === "long_afternoon_rebound"));
const luReVariant = Engine.runLunchReversal(daily, dayMinute, { shortEntryMoy: 720, flipMoy: 750, longExitMoy: 780, maxLossPoints: 50.0 });
cmpTrades("lu_variant", loadExpected("lu_variant"), luReVariant.filter(t => t.leg === "short_lunch_dip"));
cmpTrades("re_variant", loadExpected("re_variant"), luReVariant.filter(t => t.leg === "long_afternoon_rebound"));

// ---- vwap_trend ----
cmpTrades("vwap_trend_default", loadExpected("vwap_trend_default"),
  Engine.runTrendDay(daily, dayMinute, { decisionMoy: 660, minMovePoints: 20, minMovePct: null, sessionEndMoy: 805, slippagePoints: 1, minDominantSideFraction: 0.90, reference: "vwap", exitMode: "reference_break", atrWindow: 14, atrStopMult: 1.5 }));
cmpTrades("vwap_trend_open_ref", loadExpected("vwap_trend_open_ref"),
  Engine.runTrendDay(daily, dayMinute, { decisionMoy: 660, minMovePoints: 20, minMovePct: null, sessionEndMoy: 805, slippagePoints: 1, minDominantSideFraction: 0.90, reference: "open", exitMode: "reference_break", atrWindow: 14, atrStopMult: 1.5 }));
cmpTrades("vwap_trend_trailing_atr", loadExpected("vwap_trend_trailing_atr"),
  Engine.runTrendDay(daily, dayMinute, { decisionMoy: 660, minMovePoints: 20, minMovePct: null, sessionEndMoy: 805, slippagePoints: 1, minDominantSideFraction: 0.90, reference: "vwap", exitMode: "trailing_atr", atrWindow: 14, atrStopMult: 1.5 }));
cmpTrades("vwap_trend_pct", loadExpected("vwap_trend_pct"),
  Engine.runTrendDay(daily, dayMinute, { decisionMoy: 660, minMovePoints: 20, minMovePct: 0.005, sessionEndMoy: 805, slippagePoints: 1, minDominantSideFraction: 0.85, reference: "vwap", exitMode: "reference_break", atrWindow: 14, atrStopMult: 1.5 }));

// ---- night ----
cmpTrades("night_default", loadExpected("night_default"),
  Engine.runNightWindow(daily, nightMinute, { windowStartMoy: 1260, decisionMoy: 1350, windowEndMoy: 1425, minMovePoints: 20, slippagePoints: 1, minDominantSideFraction: 0.90 }));
cmpTrades("night_strict", loadExpected("night_strict"),
  Engine.runNightWindow(daily, nightMinute, { windowStartMoy: 1260, decisionMoy: 1350, windowEndMoy: 1425, minMovePoints: 20, slippagePoints: 1, minDominantSideFraction: 1.0 }));

// ---- sr_breakout / sr_fade ----
cmpTrades("sr_breakout_default", loadExpected("sr_breakout_default"),
  Engine.runSupportResistanceFade(daily, { channelWindow: 20, minRangePct: 2.0, trendMaWindow: 60, trendLookback: 20, trendSlopeThresholdPct: 5.0, stopAtrMult: 1.0, atrWindow: 14, maxHoldDays: 20, slippagePoints: 1, directionMode: "breakout" }));
cmpTrades("sr_fade_default", loadExpected("sr_fade_default"),
  Engine.runSupportResistanceFade(daily, { channelWindow: 20, minRangePct: 3.0, trendMaWindow: 60, trendLookback: 20, trendSlopeThresholdPct: 3.0, stopAtrMult: 1.0, atrWindow: 14, maxHoldDays: 20, slippagePoints: 1, directionMode: "fade" }));
cmpTrades("sr_fade_variant", loadExpected("sr_fade_variant"),
  Engine.runSupportResistanceFade(daily, { channelWindow: 15, minRangePct: 4.0, trendMaWindow: 60, trendLookback: 20, trendSlopeThresholdPct: 3.0, stopAtrMult: 1.5, atrWindow: 14, maxHoldDays: 10, slippagePoints: 1, directionMode: "fade" }));

// ---- donchian ----
cmpTrades("donchian_default", loadExpected("donchian_default"),
  Engine.runDonchian(daily, { entryWindow: 20, exitWindow: 10, atrWindow: 20, atrStopMult: 2.0, maxHoldDays: 60, slippagePoints: 1, trendFilter: false, trendMaWindow: 100, volumeFilter: false, volumeMaWindow: 20, volumeMinRatio: 1.2, volatilitySqueezeFilter: false, squeezeLookback: 60, squeezePercentile: 0.5 }));
cmpTrades("donchian_filters", loadExpected("donchian_filters"),
  Engine.runDonchian(daily, { entryWindow: 20, exitWindow: 10, atrWindow: 20, atrStopMult: 2.0, maxHoldDays: 60, slippagePoints: 1, trendFilter: true, trendMaWindow: 100, volumeFilter: true, volumeMaWindow: 20, volumeMinRatio: 1.2, volatilitySqueezeFilter: true, squeezeLookback: 60, squeezePercentile: 0.5 }));
cmpTrades("donchian_variant", loadExpected("donchian_variant"),
  Engine.runDonchian(daily, { entryWindow: 40, exitWindow: 20, atrWindow: 20, atrStopMult: 3.0, maxHoldDays: 30, slippagePoints: 1, trendFilter: false, trendMaWindow: 100, volumeFilter: false, volumeMaWindow: 20, volumeMinRatio: 1.2, volatilitySqueezeFilter: false, squeezeLookback: 60, squeezePercentile: 0.5 }));

// ---- rsi2 ----
cmpTrades("rsi2_default", loadExpected("rsi2_default"),
  Engine.runRsi2(daily, { rsiPeriod: 2, oversold: 10, overbought: 70, trendMaWindow: 200, maxHoldDays: 10, slippagePoints: 1 }));
cmpTrades("rsi2_variant", loadExpected("rsi2_variant"),
  Engine.runRsi2(daily, { rsiPeriod: 3, oversold: 15, overbought: 60, trendMaWindow: 200, maxHoldDays: 5, slippagePoints: 1 }));

// ---- orb ----
cmpTrades("orb_default", loadExpected("orb_default"),
  Engine.runORB(daily, dayMinute, { rangeStartMoy: 525, rangeEndMoy: 555, sessionEndMoy: 805, minRangePoints: 5, slippagePoints: 1, targetRMultiple: null }));
cmpTrades("orb_target", loadExpected("orb_target"),
  Engine.runORB(daily, dayMinute, { rangeStartMoy: 525, rangeEndMoy: 555, sessionEndMoy: 805, minRangePoints: 5, slippagePoints: 1, targetRMultiple: 2.0 }));

// ---- liqsweep ----
cmpTrades("liqsweep_default", loadExpected("liqsweep_default"),
  Engine.runLiquiditySweep(daily, dayMinute, { channelWindow: 20, sweepBufferPoints: 10, stopBufferPoints: 5, targetRMultiple: 2.0, maxHoldDays: 10, slippagePoints: 1 }));
cmpTrades("liqsweep_variant", loadExpected("liqsweep_variant"),
  Engine.runLiquiditySweep(daily, dayMinute, { channelWindow: 20, sweepBufferPoints: 5, stopBufferPoints: 8, targetRMultiple: 3.0, maxHoldDays: 5, slippagePoints: 1 }));

console.log(`\n=== ${totalPass}/${totalCases} cases passed ===`);
if (failures.length) {
  console.log("FAILURES:", failures.map(f => f.name).join(", "));
  process.exit(1);
}
