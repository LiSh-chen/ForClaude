/* 策略實驗室：瀏覽器端回測引擎。
 *
 * 逐行對照 tw_quant/{opening_rally,lunch_reversal,trend_day,
 * support_resistance_fade,donchian_breakout,rsi2_mean_reversion,
 * opening_range_breakout,liquidity_sweep_reversal,night_liquid_window_trend}
 * _strategy.py 移植而來，目的是讓使用者在網頁上調整參數時可以即時重新計算，
 * 不需要改Python重新匯出。每個策略函式都用 scripts/parity_dump.py 產生的
 * Python原始輸出逐筆比對過（見 test/engine_parity.test.mjs），不是憑印象
 * 重寫的近似版本。
 *
 * 已知跟Python版本的極小差異（都不影響任何實際會用到的參數組合）：
 * - trend_day/night 的「多數方向」在極端平手（正負K棒數完全相等）時，這裡
 *   固定選多方；pandas value_counts().idxmax() 的平手順序沒有文件保證，
 *   两边都算"合理"，只是選擇不一定100%一致。
 * - trend_regime_filter（目前沒有任何策略預設套用）在均線暖身期
 *   （資料開頭前 ma_window 天）的NaN處理是簡化版，實際使用中幾乎不會踩到。
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) {
    module.exports = factory();
  } else {
    root.Engine = factory();
  }
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  const POINT_VALUE = 50.0;
  const TAX_RATE_PER_SIDE = 0.00002;

  // ============================================================
  // 共用數學/序列工具（逐一對照pandas語意）
  // ============================================================

  function ewmMean(x, alpha) {
    const n = x.length;
    const y = new Float64Array(n);
    let prev = NaN, seeded = false;
    for (let i = 0; i < n; i++) {
      const xi = x[i];
      if (Number.isNaN(xi)) { y[i] = NaN; continue; }
      prev = seeded ? (alpha * xi + (1 - alpha) * prev) : xi;
      seeded = true;
      y[i] = prev;
    }
    return y;
  }

  function shift1(x) {
    const n = x.length;
    const y = new Float64Array(n);
    y[0] = NaN;
    for (let i = 1; i < n; i++) y[i] = x[i - 1];
    return y;
  }

  // pandas rolling(window).mean/max/min 預設 min_periods=window：窗內任何
  // NaN都會讓該點輸出NaN（在這份資料裡NaN只會是shift(1)造成的單一開頭缺值，
  // 不會有中間缺值）。
  function rollingMean(x, window) {
    const n = x.length;
    const y = new Float64Array(n).fill(NaN);
    let sum = 0, cnt = 0;
    for (let i = 0; i < n; i++) {
      const xi = x[i];
      if (!Number.isNaN(xi)) { sum += xi; cnt++; }
      if (i >= window) {
        const xo = x[i - window];
        if (!Number.isNaN(xo)) { sum -= xo; cnt--; }
      }
      if (i >= window - 1 && cnt >= window) y[i] = sum / window;
    }
    return y;
  }

  function rollingMax(x, window) {
    const n = x.length;
    const y = new Float64Array(n).fill(NaN);
    for (let i = window - 1; i < n; i++) {
      let mx = -Infinity, cnt = 0;
      for (let j = i - window + 1; j <= i; j++) {
        const xj = x[j];
        if (!Number.isNaN(xj)) { if (xj > mx) mx = xj; cnt++; }
      }
      if (cnt >= window) y[i] = mx;
    }
    return y;
  }

  function rollingMin(x, window) {
    const n = x.length;
    const y = new Float64Array(n).fill(NaN);
    for (let i = window - 1; i < n; i++) {
      let mn = Infinity, cnt = 0;
      for (let j = i - window + 1; j <= i; j++) {
        const xj = x[j];
        if (!Number.isNaN(xj)) { if (xj < mn) mn = xj; cnt++; }
      }
      if (cnt >= window) y[i] = mn;
    }
    return y;
  }

  // pandas Series.quantile 預設 interpolation="linear"
  function rollingQuantile(x, window, q) {
    const n = x.length;
    const y = new Float64Array(n).fill(NaN);
    for (let i = window - 1; i < n; i++) {
      const vals = [];
      for (let j = i - window + 1; j <= i; j++) {
        const xj = x[j];
        if (!Number.isNaN(xj)) vals.push(xj);
      }
      if (vals.length < window) continue;
      vals.sort((a, b) => a - b);
      const idx = q * (vals.length - 1);
      const lo = Math.floor(idx), hi = Math.ceil(idx);
      const frac = idx - lo;
      y[i] = vals[lo] + (vals[hi] - vals[lo]) * frac;
    }
    return y;
  }

  // 回傳「分數」（未乘100），呼叫端自己乘100，對齊Python的
  // `series.pct_change(n) * 100` 寫法
  function pctChangeN(x, periods) {
    const n = x.length;
    const y = new Float64Array(n).fill(NaN);
    for (let i = periods; i < n; i++) {
      const prev = x[i - periods];
      if (Number.isNaN(prev) || Number.isNaN(x[i]) || prev === 0) continue;
      y[i] = (x[i] - prev) / prev;
    }
    return y;
  }

  function computeATR(daily, window) {
    const n = daily.close.length;
    const tr = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const h = daily.high[i], l = daily.low[i];
      if (i === 0) { tr[i] = h - l; continue; }
      const pc = daily.close[i - 1];
      tr[i] = Math.max(h - l, Math.abs(h - pc), Math.abs(l - pc));
    }
    return ewmMean(tr, 1 / window);
  }

  function computeRSI(close, period) {
    const n = close.length;
    const gain = new Float64Array(n), loss = new Float64Array(n);
    gain[0] = NaN; loss[0] = NaN;
    for (let i = 1; i < n; i++) {
      const d = close[i] - close[i - 1];
      gain[i] = Math.max(d, 0);
      loss[i] = Math.max(-d, 0);
    }
    const avgGain = ewmMean(gain, 1 / period);
    const avgLoss = ewmMean(loss, 1 / period);
    const rsi = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      if (Number.isNaN(avgLoss[i]) || avgLoss[i] === 0) { rsi[i] = 100; continue; }
      const rs = avgGain[i] / avgLoss[i];
      rsi[i] = Number.isNaN(rs) ? 100 : (100 - 100 / (1 + rs));
    }
    return rsi;
  }

  // side: 'buy_stop'（做多進場/空單出場）或 'sell_stop'（做空進場/多單出場）
  function fillPriceStop(level, dayOpen, dayHigh, dayLow, side, slippage) {
    if (side === "buy_stop") {
      if (dayOpen >= level) return { price: dayOpen, reason: "gap_through" };
      return { price: level + slippage, reason: "normal_slippage" };
    }
    if (dayOpen <= level) return { price: dayOpen, reason: "gap_through" };
    return { price: level - slippage, reason: "normal_slippage" };
  }

  // side: 'buy_limit' 或 'sell_limit'
  function limitFill(level, dayOpen, side) {
    if (side === "buy_limit") return dayOpen < level ? dayOpen : level;
    return dayOpen > level ? dayOpen : level;
  }

  function applyCosts(trades, cost) {
    for (const t of trades) {
      const gross = t.pnlPoints * cost.point_value;
      const tax = cost.tax_rate_per_side * cost.point_value * (Math.abs(t.entryPrice) + Math.abs(t.exitPrice));
      const totalCost = tax + cost.commission_round_trip;
      t.grossTwd = gross;
      t.costTwd = totalCost;
      t.netTwd = gross - totalCost;
    }
    return trades;
  }

  function moyToHHMM(m) {
    const h = Math.floor(m / 60), mi = m % 60;
    return String(h).padStart(2, "0") + ":" + String(mi).padStart(2, "0");
  }
  function hhmmToMoy(s) {
    const p = s.split(":");
    return (+p[0]) * 60 + (+p[1]);
  }

  // ============================================================
  // 二進位1分K解析（見 scripts/export_strategy_lab.py write_minute_bin）
  // 格式：[numDays, (dateYYYYMMDD, barCount, [moy,o,h,l,c,v]*barCount)*numDays]
  // ============================================================
  const F_MOY = 0, F_O = 1, F_H = 2, F_L = 3, F_C = 4, F_V = 5;

  function parseMinuteBin(buf) {
    const view = new Int32Array(buf);
    let p = 0;
    const numDays = view[p++];
    const days = [];
    for (let i = 0; i < numDays; i++) {
      const dateInt = view[p++];
      const count = view[p++];
      const start = p;
      p += count * 6;
      const y = Math.floor(dateInt / 10000), m = Math.floor((dateInt % 10000) / 100), d = dateInt % 100;
      const dateStr = y + "-" + String(m).padStart(2, "0") + "-" + String(d).padStart(2, "0");
      days.push({ dateStr, view, start, count });
    }
    return days;
  }

  function barAt(block, j) {
    const b = block.start + j * 6;
    const v = block.view;
    return { moy: v[b + F_MOY], o: v[b + F_O], h: v[b + F_H], l: v[b + F_L], c: v[b + F_C], v: v[b + F_V] };
  }

  function firstBarAtOrAfter(block, moyThreshold) {
    if (!block) return null;
    for (let j = 0; j < block.count; j++) {
      if (block.view[block.start + j * 6 + F_MOY] >= moyThreshold) return j;
    }
    return null;
  }

  // ============================================================
  // 每日資料容器：從 data.json 的 records 陣列（t,o,h,l,c,v）建立
  // ============================================================
  function dailyFromRecords(records) {
    const n = records.length;
    const dateStr = new Array(n);
    const open = new Float64Array(n), high = new Float64Array(n), low = new Float64Array(n),
      close = new Float64Array(n), volume = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      const r = records[i];
      dateStr[i] = r.t;
      open[i] = r.o; high[i] = r.h; low[i] = r.l; close[i] = r.c; volume[i] = r.v;
    }
    return { n, dateStr, open, high, low, close, volume };
  }

  function buildDateIndexMap(daily) {
    const m = new Map();
    for (let i = 0; i < daily.n; i++) m.set(daily.dateStr[i], i);
    return m;
  }

  // 把逐年解析出來的分鐘K日期塊，對齊到daily的index順序上（daily.dateStr[i]
  // 對應的那一天如果有1分K資料，dayMinute[i]就是那個block，否則是null）
  function alignMinuteBlocks(daily, yearDaysArrays) {
    const map = new Map();
    for (const days of yearDaysArrays) {
      for (const d of days) map.set(d.dateStr, d);
    }
    const aligned = new Array(daily.n);
    for (let i = 0; i < daily.n; i++) aligned[i] = map.get(daily.dateStr[i]) || null;
    return aligned;
  }

  // ============================================================
  // 策略：開盤上衝 (og)
  // ============================================================
  function runOpeningRally(daily, dayMinute, cfg) {
    const trades = [];
    for (let i = 0; i < daily.n; i++) {
      const block = dayMinute[i];
      if (!block || block.count === 0) continue;
      const eIdx = firstBarAtOrAfter(block, cfg.entryMoy);
      const xIdx = firstBarAtOrAfter(block, cfg.exitMoy);
      if (eIdx === null || xIdx === null) continue;
      const eBar = barAt(block, eIdx), xBar = barAt(block, xIdx);
      if (xBar.moy <= eBar.moy) continue;
      trades.push({
        leg: "opening_rally", direction: "long", entryDate: daily.dateStr[i], entryTime: moyToHHMM(eBar.moy),
        entryPrice: eBar.o, exitDate: daily.dateStr[i], exitTime: moyToHHMM(xBar.moy), exitPrice: xBar.o,
        pnlPoints: xBar.o - eBar.o, approxTime: false,
      });
    }
    return trades;
  }

  // ============================================================
  // 策略：午盤效應 (lu=short_lunch_dip / re=long_afternoon_rebound，共用同一組參數)
  // ============================================================
  function applyProtectiveStop(block, fromIdxExclusive, toIdxInclusive, entryPrice, direction, maxLossPoints) {
    if (maxLossPoints == null) return null;
    const stopPrice = direction === "short" ? entryPrice + maxLossPoints : entryPrice - maxLossPoints;
    for (let j = fromIdxExclusive + 1; j <= toIdxInclusive; j++) {
      const bar = barAt(block, j);
      if (direction === "short" && bar.h >= stopPrice) return { price: stopPrice, moy: bar.moy };
      if (direction === "long" && bar.l <= stopPrice) return { price: stopPrice, moy: bar.moy };
    }
    return null;
  }

  function runLunchReversal(daily, dayMinute, cfg) {
    const trades = [];
    for (let i = 0; i < daily.n; i++) {
      const block = dayMinute[i];
      if (!block || block.count === 0) continue;
      const seIdx = firstBarAtOrAfter(block, cfg.shortEntryMoy);
      const flIdx = firstBarAtOrAfter(block, cfg.flipMoy);
      const lxIdx = firstBarAtOrAfter(block, cfg.longExitMoy);
      if (seIdx === null || flIdx === null || lxIdx === null) continue;
      const seBar = barAt(block, seIdx), flBar = barAt(block, flIdx), lxBar = barAt(block, lxIdx);
      if (!(seBar.moy < flBar.moy && flBar.moy < lxBar.moy)) continue;

      const shortEntryPrice = seBar.o;
      const stopHit1 = applyProtectiveStop(block, seIdx, flIdx, shortEntryPrice, "short", cfg.maxLossPoints);
      const sp = stopHit1 ? stopHit1.price : flBar.o;
      const sMoy = stopHit1 ? stopHit1.moy : flBar.moy;
      trades.push({
        leg: "short_lunch_dip", direction: "short", entryDate: daily.dateStr[i], entryTime: moyToHHMM(seBar.moy),
        entryPrice: shortEntryPrice, exitDate: daily.dateStr[i], exitTime: moyToHHMM(sMoy), exitPrice: sp,
        pnlPoints: shortEntryPrice - sp, approxTime: false,
      });

      const longEntryPrice = flBar.o;
      const stopHit2 = applyProtectiveStop(block, flIdx, lxIdx, longEntryPrice, "long", cfg.maxLossPoints);
      const lp = stopHit2 ? stopHit2.price : lxBar.o;
      const lMoy = stopHit2 ? stopHit2.moy : lxBar.moy;
      trades.push({
        leg: "long_afternoon_rebound", direction: "long", entryDate: daily.dateStr[i], entryTime: moyToHHMM(flBar.moy),
        entryPrice: longEntryPrice, exitDate: daily.dateStr[i], exitTime: moyToHHMM(lMoy), exitPrice: lp,
        pnlPoints: lp - longEntryPrice, approxTime: false,
      });
    }
    return trades;
  }

  // ============================================================
  // 共用：時段內累積VWAP（trend_day / night_liquid_window 共用）
  // ============================================================
  function cumulativeVWAP(block) {
    const n = block.count;
    const vwap = new Float64Array(n);
    let cumPV = 0, cumV = 0;
    for (let j = 0; j < n; j++) {
      const bar = barAt(block, j);
      const tp = (bar.h + bar.l + bar.c) / 3;
      cumPV += tp * bar.v; cumV += bar.v;
      vwap[j] = cumV > 0 ? cumPV / cumV : NaN;
    }
    return vwap;
  }

  // ============================================================
  // 策略：VWAP趨勢日 (vwap_trend / trend_day_strategy.py)
  // ============================================================
  function runTrendDay(daily, dayMinute, cfg) {
    const trades = [];
    let atrPrior = null;
    if (cfg.exitMode === "trailing_atr") {
      const atr = computeATR(daily, cfg.atrWindow);
      atrPrior = shift1(atr);
    }
    for (let i = 0; i < daily.n; i++) {
      const block = dayMinute[i];
      if (!block || block.count === 0) continue;
      const n = block.count;
      const refPrice = cfg.reference === "vwap" ? cumulativeVWAP(block) : null;
      const openRef = cfg.reference === "vwap" ? null : barAt(block, 0).o;
      function ref(j) { return cfg.reference === "vwap" ? refPrice[j] : openRef; }

      let decisionIdx = -1;
      for (let j = 0; j < n; j++) { if (barAt(block, j).moy <= cfg.decisionMoy) decisionIdx = j; else break; }
      if (decisionIdx === -1 || decisionIdx + 1 >= n) continue;

      let posCount = 0, negCount = 0;
      for (let j = 0; j <= decisionIdx; j++) {
        const diff = barAt(block, j).c - ref(j);
        if (diff > 0) posCount++; else if (diff < 0) negCount++;
      }
      const total = posCount + negCount;
      if (total === 0) continue;
      const dominantSign = posCount >= negCount ? 1 : -1;
      const dominantFraction = (posCount >= negCount ? posCount : negCount) / total;
      if (dominantFraction < cfg.minDominantSideFraction) continue;

      const dayOpen = barAt(block, 0).o;
      const decisionBar = barAt(block, decisionIdx);
      const move = (decisionBar.c - dayOpen) * dominantSign;
      const moveThreshold = cfg.minMovePct != null ? cfg.minMovePct * dayOpen : cfg.minMovePoints;
      if (move < moveThreshold) continue;

      if (cfg.exitMode === "trailing_atr") {
        const ap = atrPrior[i];
        if (Number.isNaN(ap)) continue;
      }

      const direction = dominantSign > 0 ? "long" : "short";
      const entryBar = barAt(block, decisionIdx + 1);
      const entryPrice = entryBar.o + (direction === "long" ? cfg.slippagePoints : -cfg.slippagePoints);

      let exitPrice = null, exitMoy = null;
      if (cfg.exitMode === "trailing_atr") {
        const trailDistance = cfg.atrStopMult * atrPrior[i];
        let extreme = entryPrice;
        for (let j = decisionIdx + 1; j < n; j++) {
          const bar = barAt(block, j);
          if (bar.moy >= cfg.sessionEndMoy) break;
          if (direction === "long") {
            extreme = Math.max(extreme, bar.c);
            if (bar.c <= extreme - trailDistance) { exitPrice = bar.c - cfg.slippagePoints; exitMoy = bar.moy; break; }
          } else {
            extreme = Math.min(extreme, bar.c);
            if (bar.c >= extreme + trailDistance) { exitPrice = bar.c + cfg.slippagePoints; exitMoy = bar.moy; break; }
          }
        }
      } else {
        for (let j = decisionIdx + 1; j < n; j++) {
          const bar = barAt(block, j);
          if (bar.moy >= cfg.sessionEndMoy) break;
          const broke = direction === "long" ? (bar.c < ref(j)) : (bar.c > ref(j));
          if (broke) { exitPrice = bar.c + (direction === "long" ? -cfg.slippagePoints : cfg.slippagePoints); exitMoy = bar.moy; break; }
        }
      }
      if (exitPrice === null) {
        let lastIdx = -1;
        for (let j = 0; j < n; j++) { if (barAt(block, j).moy >= cfg.sessionEndMoy) { lastIdx = j; break; } }
        const lastBar = lastIdx >= 0 ? barAt(block, lastIdx) : barAt(block, n - 1);
        exitPrice = lastBar.o + (direction === "long" ? -cfg.slippagePoints : cfg.slippagePoints);
        exitMoy = lastBar.moy;
      }

      const pnlPoints = direction === "long" ? (exitPrice - entryPrice) : (entryPrice - exitPrice);
      trades.push({
        direction, entryDate: daily.dateStr[i], entryTime: moyToHHMM(entryBar.moy), entryPrice,
        exitDate: daily.dateStr[i], exitTime: moyToHHMM(exitMoy), exitPrice, pnlPoints, approxTime: false,
      });
    }
    return trades;
  }

  // ============================================================
  // 策略：夜盤流動性熱區 (night_liquid_window_trend_strategy.py)
  // ============================================================
  function runNightWindow(daily, nightMinute, cfg) {
    const trades = [];
    for (let i = 0; i < daily.n; i++) {
      const block = nightMinute[i];
      if (!block || block.count === 0) continue;
      const n = block.count;
      const refPrice = cumulativeVWAP(block);

      let decisionIdx = -1;
      for (let j = 0; j < n; j++) { if (barAt(block, j).moy <= cfg.decisionMoy) decisionIdx = j; else break; }
      if (decisionIdx === -1 || decisionIdx + 1 >= n) continue;

      let posCount = 0, negCount = 0;
      for (let j = 0; j <= decisionIdx; j++) {
        const diff = barAt(block, j).c - refPrice[j];
        if (diff > 0) posCount++; else if (diff < 0) negCount++;
      }
      const total = posCount + negCount;
      if (total === 0) continue;
      const dominantSign = posCount >= negCount ? 1 : -1;
      const dominantFraction = (posCount >= negCount ? posCount : negCount) / total;
      if (dominantFraction < cfg.minDominantSideFraction) continue;

      const windowOpen = barAt(block, 0).o;
      const decisionBar = barAt(block, decisionIdx);
      const move = (decisionBar.c - windowOpen) * dominantSign;
      if (move < cfg.minMovePoints) continue;

      const direction = dominantSign > 0 ? "long" : "short";
      const entryBar = barAt(block, decisionIdx + 1);
      const entryPrice = entryBar.o + (direction === "long" ? cfg.slippagePoints : -cfg.slippagePoints);

      let exitPrice = null, exitMoy = null;
      for (let j = decisionIdx + 1; j < n; j++) {
        const bar = barAt(block, j);
        if (bar.moy >= cfg.windowEndMoy) break;
        const broke = direction === "long" ? (bar.c < refPrice[j]) : (bar.c > refPrice[j]);
        if (broke) { exitPrice = bar.c + (direction === "long" ? -cfg.slippagePoints : cfg.slippagePoints); exitMoy = bar.moy; break; }
      }
      if (exitPrice === null) {
        let lastIdx = -1;
        for (let j = 0; j < n; j++) { if (barAt(block, j).moy >= cfg.windowEndMoy) { lastIdx = j; break; } }
        const lastBar = lastIdx >= 0 ? barAt(block, lastIdx) : barAt(block, n - 1);
        exitPrice = lastBar.o + (direction === "long" ? -cfg.slippagePoints : cfg.slippagePoints);
        exitMoy = lastBar.moy;
      }

      const pnlPoints = direction === "long" ? (exitPrice - entryPrice) : (entryPrice - exitPrice);
      trades.push({
        direction, entryDate: daily.dateStr[i], entryTime: moyToHHMM(entryBar.moy), entryPrice,
        exitDate: daily.dateStr[i], exitTime: moyToHHMM(exitMoy), exitPrice, pnlPoints, approxTime: false,
      });
    }
    return trades;
  }

  // ============================================================
  // 策略：支撐壓力 fade/breakout (support_resistance_fade_strategy.py)
  // ============================================================
  function runSupportResistanceFade(daily, cfg) {
    const n = daily.n;
    const atr = computeATR(daily, cfg.atrWindow);
    const highShift1 = shift1(daily.high), lowShift1 = shift1(daily.low), closeShift1 = shift1(daily.close);
    const resistance = rollingMax(highShift1, cfg.channelWindow);
    const support = rollingMin(lowShift1, cfg.channelWindow);
    const rangePct = new Float64Array(n);
    for (let i = 0; i < n; i++) rangePct[i] = (resistance[i] - support[i]) / closeShift1[i] * 100;
    const trendMa = rollingMean(closeShift1, cfg.trendMaWindow);
    const trendSlopeFrac = pctChangeN(trendMa, cfg.trendLookback);

    const trades = [];
    let i = 0;
    while (i < n) {
      if (Number.isNaN(support[i]) || Number.isNaN(trendSlopeFrac[i]) || Number.isNaN(atr[i])) { i++; continue; }
      const trendSlopePct = trendSlopeFrac[i] * 100;
      if (rangePct[i] < cfg.minRangePct || Math.abs(trendSlopePct) > cfg.trendSlopeThresholdPct) { i++; continue; }

      let direction = null, entryPrice, target = null, stop;
      const open = daily.open[i], high = daily.high[i], low = daily.low[i];
      if (cfg.directionMode === "fade") {
        if (low <= support[i]) {
          direction = "long"; entryPrice = limitFill(support[i], open, "buy_limit");
          target = resistance[i]; stop = entryPrice - cfg.stopAtrMult * atr[i];
        } else if (high >= resistance[i]) {
          direction = "short"; entryPrice = limitFill(resistance[i], open, "sell_limit");
          target = support[i]; stop = entryPrice + cfg.stopAtrMult * atr[i];
        }
      } else {
        if (high >= resistance[i]) {
          direction = "long";
          const f = fillPriceStop(resistance[i], open, high, low, "buy_stop", cfg.slippagePoints);
          entryPrice = f.price; stop = entryPrice - cfg.stopAtrMult * atr[i];
        } else if (low <= support[i]) {
          direction = "short";
          const f = fillPriceStop(support[i], open, high, low, "sell_stop", cfg.slippagePoints);
          entryPrice = f.price; stop = entryPrice + cfg.stopAtrMult * atr[i];
        }
      }
      if (direction === null) { i++; continue; }

      const entryDate = daily.dateStr[i];
      let exitPrice = null, exitIdx = null;
      const jLimit = Math.min(i + cfg.maxHoldDays, n - 1);
      let j = i + 1;
      while (j <= jLimit) {
        const rh = daily.high[j], rl = daily.low[j], ro = daily.open[j];
        if (direction === "long") {
          if (rl <= stop) { const f = fillPriceStop(stop, ro, rh, rl, "sell_stop", cfg.slippagePoints); exitPrice = f.price; exitIdx = j; break; }
          if (target !== null && rh >= target) { exitPrice = limitFill(target, ro, "sell_limit"); exitIdx = j; break; }
        } else {
          if (rh >= stop) { const f = fillPriceStop(stop, ro, rh, rl, "buy_stop", cfg.slippagePoints); exitPrice = f.price; exitIdx = j; break; }
          if (target !== null && rl <= target) { exitPrice = limitFill(target, ro, "buy_limit"); exitIdx = j; break; }
        }
        j++;
      }
      if (exitPrice === null) {
        exitIdx = jLimit;
        exitPrice = direction === "long" ? daily.close[jLimit] - cfg.slippagePoints : daily.close[jLimit] + cfg.slippagePoints;
      }
      const pnlPoints = direction === "long" ? (exitPrice - entryPrice) : (entryPrice - exitPrice);
      trades.push({
        direction, entryDate, entryTime: "08:45", entryPrice,
        exitDate: daily.dateStr[exitIdx], exitTime: "13:30", exitPrice, pnlPoints, approxTime: true,
      });
      i = exitIdx + 1;
    }
    return trades;
  }

  // ============================================================
  // 策略：唐奇安突破 (donchian_breakout_strategy.py)
  // ============================================================
  function runDonchian(daily, cfg) {
    const n = daily.n;
    const atr = computeATR(daily, cfg.atrWindow);
    const highShift1 = shift1(daily.high), lowShift1 = shift1(daily.low), closeShift1 = shift1(daily.close), volShift1 = shift1(daily.volume);
    const upperEntry = rollingMax(highShift1, cfg.entryWindow);
    const lowerEntry = rollingMin(lowShift1, cfg.entryWindow);
    const upperExitRef = rollingMax(highShift1, cfg.exitWindow);
    const lowerExitRef = rollingMin(lowShift1, cfg.exitWindow);
    const trendMaPrev = rollingMean(closeShift1, cfg.trendMaWindow);
    const volumeMaPrev = rollingMean(volShift1, cfg.volumeMaWindow);
    const atrRatio = new Float64Array(n);
    for (let i = 0; i < n; i++) atrRatio[i] = atr[i] / daily.close[i];
    const atrRatioThresholdPrev = rollingQuantile(shift1(atrRatio), cfg.squeezeLookback, cfg.squeezePercentile);

    function passesFilters(i, direction) {
      if (cfg.trendFilter) {
        if (Number.isNaN(trendMaPrev[i])) return false;
        if (direction === "long" && !(closeShift1[i] > trendMaPrev[i])) return false;
        if (direction === "short" && !(closeShift1[i] < trendMaPrev[i])) return false;
      }
      if (cfg.volumeFilter) {
        if (Number.isNaN(volumeMaPrev[i]) || volumeMaPrev[i] <= 0) return false;
        if (daily.volume[i] < cfg.volumeMinRatio * volumeMaPrev[i]) return false;
      }
      if (cfg.volatilitySqueezeFilter) {
        if (Number.isNaN(atrRatioThresholdPrev[i])) return false;
        if (atrRatio[i] >= atrRatioThresholdPrev[i]) return false;
      }
      return true;
    }

    const trades = [];
    let i = 0;
    while (i < n) {
      if (Number.isNaN(upperEntry[i]) || Number.isNaN(atr[i])) { i++; continue; }
      let direction = null, entryLevel;
      if (daily.high[i] >= upperEntry[i]) { direction = "long"; entryLevel = upperEntry[i]; }
      else if (daily.low[i] <= lowerEntry[i]) { direction = "short"; entryLevel = lowerEntry[i]; }
      if (direction === null) { i++; continue; }
      if (!passesFilters(i, direction)) { i++; continue; }

      const side = direction === "long" ? "buy_stop" : "sell_stop";
      const f = fillPriceStop(entryLevel, daily.open[i], daily.high[i], daily.low[i], side, cfg.slippagePoints);
      const entryPrice = f.price;
      const entryDate = daily.dateStr[i];
      let stop = direction === "long" ? entryPrice - cfg.atrStopMult * atr[i] : entryPrice + cfg.atrStopMult * atr[i];

      let exitPrice = null, exitIdx = null;
      const jLimit = Math.min(i + cfg.maxHoldDays, n - 1);
      let j = i + 1;
      while (j <= jLimit) {
        if (direction === "long") {
          if (!Number.isNaN(lowerExitRef[j])) stop = Math.max(stop, lowerExitRef[j]);
          if (daily.low[j] <= stop) {
            const ff = fillPriceStop(stop, daily.open[j], daily.high[j], daily.low[j], "sell_stop", cfg.slippagePoints);
            exitPrice = ff.price; exitIdx = j; break;
          }
        } else {
          if (!Number.isNaN(upperExitRef[j])) stop = Math.min(stop, upperExitRef[j]);
          if (daily.high[j] >= stop) {
            const ff = fillPriceStop(stop, daily.open[j], daily.high[j], daily.low[j], "buy_stop", cfg.slippagePoints);
            exitPrice = ff.price; exitIdx = j; break;
          }
        }
        j++;
      }
      if (exitPrice === null) { exitIdx = jLimit; exitPrice = daily.close[jLimit]; }
      const pnlPoints = direction === "long" ? (exitPrice - entryPrice) : (entryPrice - exitPrice);
      trades.push({
        direction, entryDate, entryTime: "08:45", entryPrice,
        exitDate: daily.dateStr[exitIdx], exitTime: "13:30", exitPrice, pnlPoints, approxTime: true,
      });
      i = exitIdx + 1;
    }
    return trades;
  }

  // ============================================================
  // 策略：RSI2均值回歸 (rsi2_mean_reversion_strategy.py)
  // ============================================================
  function runRsi2(daily, cfg) {
    const n = daily.n;
    const rsi = computeRSI(daily.close, cfg.rsiPeriod);
    const trendMa = rollingMean(daily.close, cfg.trendMaWindow); // 注意：沒有shift(1)，跟Python一致

    const trades = [];
    let i = 0;
    while (i < n - 1) {
      if (Number.isNaN(trendMa[i]) || Number.isNaN(rsi[i])) { i++; continue; }
      let direction = null;
      if (daily.close[i] > trendMa[i] && rsi[i] < cfg.oversold) direction = "long";
      else if (daily.close[i] < trendMa[i] && rsi[i] > (100 - cfg.oversold)) direction = "short";
      if (direction === null) { i++; continue; }

      const entryIdx = i + 1;
      const entryPrice = direction === "long" ? daily.open[entryIdx] + cfg.slippagePoints : daily.open[entryIdx] - cfg.slippagePoints;

      let exitIdx = null;
      const jLimit = Math.min(entryIdx + cfg.maxHoldDays, n - 1);
      let j = entryIdx;
      while (j < jLimit) {
        const exitSignal = (direction === "long" && rsi[j] > cfg.overbought) || (direction === "short" && rsi[j] < (100 - cfg.overbought));
        if (exitSignal) { exitIdx = j + 1; break; }
        j++;
      }
      if (exitIdx === null) exitIdx = jLimit;

      const exitPrice = direction === "long" ? daily.open[exitIdx] - cfg.slippagePoints : daily.open[exitIdx] + cfg.slippagePoints;
      const pnlPoints = direction === "long" ? (exitPrice - entryPrice) : (entryPrice - exitPrice);
      trades.push({
        direction, entryDate: daily.dateStr[entryIdx], entryTime: "08:45", entryPrice,
        exitDate: daily.dateStr[exitIdx], exitTime: "13:30", exitPrice, pnlPoints, approxTime: true,
      });
      i = exitIdx;
    }
    return trades;
  }

  // ============================================================
  // 策略：開盤區間突破 ORB (opening_range_breakout_strategy.py)
  // ============================================================
  function runORB(daily, dayMinute, cfg) {
    const trades = [];
    for (let i = 0; i < daily.n; i++) {
      const block = dayMinute[i];
      if (!block || block.count === 0) continue;
      const n = block.count;

      let rangeHigh = -Infinity, rangeLow = Infinity, haveRange = false;
      for (let j = 0; j < n; j++) {
        const bar = barAt(block, j);
        if (bar.moy >= cfg.rangeStartMoy && bar.moy < cfg.rangeEndMoy) {
          rangeHigh = Math.max(rangeHigh, bar.h); rangeLow = Math.min(rangeLow, bar.l); haveRange = true;
        }
      }
      if (!haveRange) continue;
      if (rangeHigh - rangeLow < cfg.minRangePoints) continue;

      const after = [];
      for (let j = 0; j < n; j++) {
        const bar = barAt(block, j);
        if (bar.moy >= cfg.rangeEndMoy && bar.moy < cfg.sessionEndMoy) after.push(j);
      }
      if (after.length === 0) continue;

      let longIdx = null, shortIdx = null;
      for (let k = 0; k < after.length; k++) {
        const bar = barAt(block, after[k]);
        if (longIdx === null && bar.h >= rangeHigh) longIdx = k;
        if (shortIdx === null && bar.l <= rangeLow) shortIdx = k;
      }
      if (longIdx === null && shortIdx === null) continue;
      if (longIdx !== null && shortIdx !== null && longIdx === shortIdx) continue;

      let direction, entryK, entryLevel, stopLevel;
      if (shortIdx === null || (longIdx !== null && longIdx < shortIdx)) {
        direction = "long"; entryK = longIdx; entryLevel = rangeHigh; stopLevel = rangeLow;
      } else {
        direction = "short"; entryK = shortIdx; entryLevel = rangeLow; stopLevel = rangeHigh;
      }

      const entryBar = barAt(block, after[entryK]);
      const side = direction === "long" ? "buy_stop" : "sell_stop";
      const f = fillPriceStop(entryLevel, entryBar.o, entryBar.h, entryBar.l, side, cfg.slippagePoints);
      const entryPrice = f.price;

      let target = null;
      if (cfg.targetRMultiple != null) {
        const risk = Math.abs(entryPrice - stopLevel);
        target = direction === "long" ? entryPrice + cfg.targetRMultiple * risk : entryPrice - cfg.targetRMultiple * risk;
      }

      let exitPrice = null, exitMoy = null;
      for (let k = entryK + 1; k < after.length; k++) {
        const bar = barAt(block, after[k]);
        if (direction === "long") {
          if (bar.l <= stopLevel) { const ff = fillPriceStop(stopLevel, bar.o, bar.h, bar.l, "sell_stop", cfg.slippagePoints); exitPrice = ff.price; exitMoy = bar.moy; break; }
          if (target !== null && bar.h >= target) { exitPrice = target; exitMoy = bar.moy; break; }
        } else {
          if (bar.h >= stopLevel) { const ff = fillPriceStop(stopLevel, bar.o, bar.h, bar.l, "buy_stop", cfg.slippagePoints); exitPrice = ff.price; exitMoy = bar.moy; break; }
          if (target !== null && bar.l <= target) { exitPrice = target; exitMoy = bar.moy; break; }
        }
      }
      if (exitPrice === null) {
        let lastIdx = -1;
        for (let j = 0; j < n; j++) { if (barAt(block, j).moy >= cfg.sessionEndMoy) { lastIdx = j; break; } }
        const lastBar = lastIdx >= 0 ? barAt(block, lastIdx) : barAt(block, n - 1);
        exitPrice = lastBar.o + (direction === "long" ? -cfg.slippagePoints : cfg.slippagePoints);
        exitMoy = lastBar.moy;
      }
      const pnlPoints = direction === "long" ? (exitPrice - entryPrice) : (entryPrice - exitPrice);
      trades.push({
        direction, entryDate: daily.dateStr[i], entryTime: moyToHHMM(entryBar.moy), entryPrice,
        exitDate: daily.dateStr[i], exitTime: moyToHHMM(exitMoy), exitPrice, pnlPoints, approxTime: false,
      });
    }
    return trades;
  }

  // ============================================================
  // 策略：ICT流動性掃單 (liquidity_sweep_reversal_strategy.py)
  // ============================================================
  function runLiquiditySweep(daily, dayMinute, cfg) {
    const n = daily.n;
    const highShift1 = shift1(daily.high), lowShift1 = shift1(daily.low);
    const resistance = rollingMax(highShift1, cfg.channelWindow);
    const support = rollingMin(lowShift1, cfg.channelWindow);

    const trades = [];
    let i = 0;
    while (i < n) {
      if (Number.isNaN(support[i]) || Number.isNaN(resistance[i])) { i++; continue; }
      const block = dayMinute[i];
      if (!block || block.count < 3) { i++; continue; }
      const m = block.count;

      let runningLow = Infinity, runningHigh = -Infinity;
      let longIdx = null, shortIdx = null;
      for (let j = 0; j < m; j++) {
        const bar = barAt(block, j);
        runningLow = Math.min(runningLow, bar.l);
        runningHigh = Math.max(runningHigh, bar.h);
        const sweptSupport = runningLow <= (support[i] - cfg.sweepBufferPoints);
        const sweptResistance = runningHigh >= (resistance[i] + cfg.sweepBufferPoints);
        if (longIdx === null && sweptSupport && bar.c > support[i]) longIdx = j;
        if (shortIdx === null && sweptResistance && bar.c < resistance[i]) shortIdx = j;
      }
      let signalIdx, direction;
      if (longIdx !== null && (shortIdx === null || longIdx <= shortIdx)) { signalIdx = longIdx; direction = "long"; }
      else if (shortIdx !== null) { signalIdx = shortIdx; direction = "short"; }
      else { i++; continue; }

      if (signalIdx + 1 >= m) { i++; continue; }
      const entryBar = barAt(block, signalIdx + 1);
      const entryPrice = entryBar.o + (direction === "long" ? cfg.slippagePoints : -cfg.slippagePoints);

      let sweepExtreme = direction === "long" ? Infinity : -Infinity;
      for (let j = 0; j <= signalIdx; j++) {
        const bar = barAt(block, j);
        sweepExtreme = direction === "long" ? Math.min(sweepExtreme, bar.l) : Math.max(sweepExtreme, bar.h);
      }
      const stop0 = direction === "long" ? sweepExtreme - cfg.stopBufferPoints : sweepExtreme + cfg.stopBufferPoints;
      if ((direction === "long" && entryPrice <= stop0) || (direction === "short" && entryPrice >= stop0)) { i++; continue; }

      const risk = Math.abs(entryPrice - stop0);
      const target = direction === "long" ? entryPrice + cfg.targetRMultiple * risk : entryPrice - cfg.targetRMultiple * risk;

      let exitPrice = null, exitMoy = null, usedMinuteExit = false;
      for (let j = signalIdx + 2; j < m; j++) {
        const bar = barAt(block, j);
        if (direction === "long") {
          if (bar.l <= stop0) { const f = fillPriceStop(stop0, bar.o, bar.h, bar.l, "sell_stop", cfg.slippagePoints); exitPrice = f.price; exitMoy = bar.moy; usedMinuteExit = true; break; }
          if (bar.h >= target) { exitPrice = limitFill(target, bar.o, "sell_limit"); exitMoy = bar.moy; usedMinuteExit = true; break; }
        } else {
          if (bar.h >= stop0) { const f = fillPriceStop(stop0, bar.o, bar.h, bar.l, "buy_stop", cfg.slippagePoints); exitPrice = f.price; exitMoy = bar.moy; usedMinuteExit = true; break; }
          if (bar.l <= target) { exitPrice = limitFill(target, bar.o, "buy_limit"); exitMoy = bar.moy; usedMinuteExit = true; break; }
        }
      }

      let exitDateIdx = i, jDateIdx = i;
      if (exitPrice === null) {
        let holdDaysUsed = 0;
        while (exitPrice === null && holdDaysUsed < cfg.maxHoldDays) {
          jDateIdx++;
          if (jDateIdx >= n) break;
          holdDaysUsed++;
          const rh = daily.high[jDateIdx], rl = daily.low[jDateIdx], ro = daily.open[jDateIdx];
          if (direction === "long") {
            if (rl <= stop0) { const f = fillPriceStop(stop0, ro, rh, rl, "sell_stop", cfg.slippagePoints); exitPrice = f.price; exitDateIdx = jDateIdx; break; }
            if (rh >= target) { exitPrice = limitFill(target, ro, "sell_limit"); exitDateIdx = jDateIdx; break; }
          } else {
            if (rh >= stop0) { const f = fillPriceStop(stop0, ro, rh, rl, "buy_stop", cfg.slippagePoints); exitPrice = f.price; exitDateIdx = jDateIdx; break; }
            if (rl <= target) { exitPrice = limitFill(target, ro, "buy_limit"); exitDateIdx = jDateIdx; break; }
          }
        }
      } else {
        exitDateIdx = i;
      }

      if (exitPrice === null) {
        const lastDateIdx = Math.min(i + cfg.maxHoldDays, n - 1);
        exitPrice = direction === "long" ? daily.close[lastDateIdx] - cfg.slippagePoints : daily.close[lastDateIdx] + cfg.slippagePoints;
        exitDateIdx = lastDateIdx; jDateIdx = lastDateIdx;
      }

      const approxTime = !usedMinuteExit;
      const exitTime = usedMinuteExit ? moyToHHMM(exitMoy) : "13:30";
      const pnlPoints = direction === "long" ? (exitPrice - entryPrice) : (entryPrice - exitPrice);
      trades.push({
        direction, entryDate: daily.dateStr[i], entryTime: moyToHHMM(entryBar.moy), entryPrice,
        exitDate: daily.dateStr[exitDateIdx], exitTime, exitPrice, pnlPoints, approxTime,
      });
      i = jDateIdx + 1;
    }
    return trades;
  }

  // ============================================================
  // 濾網：量能（re腿用固定IS門檻）、200日均線趨勢對齊（選用）
  // ============================================================
  function computeVolRatioLag1(daily) {
    const volMean20 = rollingMean(daily.volume, 20);
    const volRatio = new Float64Array(daily.n);
    for (let i = 0; i < daily.n; i++) volRatio[i] = daily.volume[i] / volMean20[i];
    return shift1(volRatio);
  }

  function applyVolumeFilterFixed(trades, daily, dateIndexMap, threshold) {
    const volRatioLag1 = computeVolRatioLag1(daily);
    return trades.filter((t) => {
      const idx = dateIndexMap.get(t.entryDate);
      if (idx === undefined) return false;
      const val = volRatioLag1[idx];
      return !Number.isNaN(val) && val >= threshold;
    });
  }

  function applyTrendRegimeFilter(trades, daily, dateIndexMap, spec) {
    const closeShift1 = shift1(daily.close);
    const ma = rollingMean(closeShift1, spec.maWindow);
    const regimeLong = new Array(daily.n).fill(null);
    for (let i = 0; i < daily.n; i++) {
      if (!Number.isNaN(closeShift1[i]) && !Number.isNaN(ma[i])) regimeLong[i] = closeShift1[i] > ma[i];
    }
    return trades.filter((t) => {
      const idx = dateIndexMap.get(t.entryDate);
      const regime = idx === undefined ? null : regimeLong[idx];
      const isLong = t.direction === "long";
      const aligned = regime === null ? false : (isLong ? regime : !regime);
      return spec.alignedOnly ? aligned : !aligned;
    });
  }

  function combine(tradeArrays) {
    const out = [];
    for (const arr of tradeArrays) for (const t of arr) out.push(t);
    out.sort((a, b) => {
      const ka = a.entryDate + a.entryTime, kb = b.entryDate + b.entryTime;
      return ka < kb ? -1 : ka > kb ? 1 : 0;
    });
    return out;
  }

  // ============================================================
  // 每個策略的可調參數 schema（預設值對照 tw_quant/strategy_lab.py build_registry()
  // 目前註冊的鎖定參數，也就是 scripts/export_strategy_lab.py 產生初始畫面用的那組）
  // type: 'time'（分鐘數，UI用HH:MM輸入）| 'number' | 'bool' | 'nullable_number'
  // ============================================================
  const STRATEGY_PARAMS = {
    og: [
      { key: "entryMoy", label: "進場時間", type: "time", default: 525 },
      { key: "exitMoy", label: "出場時間", type: "time", default: 540 },
    ],
    lu_re: [
      { key: "shortEntryMoy", label: "放空進場時間", type: "time", default: 720 },
      { key: "flipMoy", label: "翻多時間", type: "time", default: 750 },
      { key: "longExitMoy", label: "多單出場時間", type: "time", default: 780 },
      { key: "maxLossPoints", label: "保護性停損(點，留空=不設)", type: "nullable_number", default: null, min: 1, max: 500, step: 1 },
    ],
    re_filter: [
      { key: "applyVolumeFilter", label: "套用量能濾網(固定IS門檻，原始設計)", type: "bool", default: true },
    ],
    vwap_trend: [
      { key: "decisionMoy", label: "決策時間", type: "time", default: 660 },
      { key: "minMovePoints", label: "最小幅度(點)", type: "number", default: 20, min: 0, max: 200, step: 1 },
      { key: "sessionEndMoy", label: "收盤前強制平倉", type: "time", default: 805 },
      { key: "slippagePoints", label: "滑價(點)", type: "number", default: 1, min: 0, max: 10, step: 0.5 },
      { key: "minDominantSideFraction", label: "多數方向門檻(1.0=完全不回頭)", type: "number", default: 0.90, min: 0.5, max: 1.0, step: 0.01 },
      { key: "reference", label: "參考水準", type: "select", default: "vwap", options: [["vwap", "VWAP(動態)"], ["open", "開盤價(固定)"]] },
      { key: "exitMode", label: "出場機制", type: "select", default: "reference_break", options: [["reference_break", "穿越參考水準"], ["trailing_atr", "ATR移動停損"]] },
      { key: "atrWindow", label: "ATR天數(trailing_atr用)", type: "number", default: 14, min: 2, max: 60, step: 1 },
      { key: "atrStopMult", label: "ATR停損倍數(trailing_atr用)", type: "number", default: 1.5, min: 0.5, max: 5, step: 0.1 },
    ],
    sr_breakout: [
      { key: "channelWindow", label: "通道天數", type: "number", default: 20, min: 5, max: 120, step: 1 },
      { key: "minRangePct", label: "最小通道寬度(%)", type: "number", default: 2.0, min: 0, max: 20, step: 0.1 },
      { key: "trendMaWindow", label: "趨勢均線天數", type: "number", default: 60, min: 5, max: 250, step: 1 },
      { key: "trendLookback", label: "斜率回看天數", type: "number", default: 20, min: 2, max: 120, step: 1 },
      { key: "trendSlopeThresholdPct", label: "斜率門檻(%)", type: "number", default: 5.0, min: 0, max: 30, step: 0.1 },
      { key: "stopAtrMult", label: "停損ATR倍數", type: "number", default: 1.0, min: 0.1, max: 5, step: 0.1 },
      { key: "atrWindow", label: "ATR天數", type: "number", default: 14, min: 2, max: 60, step: 1 },
      { key: "maxHoldDays", label: "最長持有天數", type: "number", default: 20, min: 1, max: 120, step: 1 },
      { key: "slippagePoints", label: "滑價(點)", type: "number", default: 1, min: 0, max: 10, step: 0.5 },
    ],
    sr_fade: [
      { key: "channelWindow", label: "通道天數", type: "number", default: 20, min: 5, max: 120, step: 1 },
      { key: "minRangePct", label: "最小通道寬度(%)", type: "number", default: 3.0, min: 0, max: 20, step: 0.1 },
      { key: "trendMaWindow", label: "趨勢均線天數", type: "number", default: 60, min: 5, max: 250, step: 1 },
      { key: "trendLookback", label: "斜率回看天數", type: "number", default: 20, min: 2, max: 120, step: 1 },
      { key: "trendSlopeThresholdPct", label: "斜率門檻(%)", type: "number", default: 3.0, min: 0, max: 30, step: 0.1 },
      { key: "stopAtrMult", label: "停損ATR倍數", type: "number", default: 1.0, min: 0.1, max: 5, step: 0.1 },
      { key: "atrWindow", label: "ATR天數", type: "number", default: 14, min: 2, max: 60, step: 1 },
      { key: "maxHoldDays", label: "最長持有天數", type: "number", default: 20, min: 1, max: 120, step: 1 },
      { key: "slippagePoints", label: "滑價(點)", type: "number", default: 1, min: 0, max: 10, step: 0.5 },
    ],
    donchian: [
      { key: "entryWindow", label: "進場通道天數", type: "number", default: 20, min: 5, max: 120, step: 1 },
      { key: "exitWindow", label: "出場通道天數", type: "number", default: 10, min: 2, max: 120, step: 1 },
      { key: "atrWindow", label: "ATR天數", type: "number", default: 20, min: 2, max: 60, step: 1 },
      { key: "atrStopMult", label: "初始停損ATR倍數", type: "number", default: 2.0, min: 0.5, max: 6, step: 0.1 },
      { key: "maxHoldDays", label: "最長持有天數", type: "number", default: 60, min: 1, max: 250, step: 1 },
      { key: "slippagePoints", label: "滑價(點)", type: "number", default: 1, min: 0, max: 10, step: 0.5 },
      { key: "trendFilter", label: "趨勢濾網(要求跟長均線同向)", type: "bool", default: false },
      { key: "trendMaWindow", label: "趨勢均線天數", type: "number", default: 100, min: 5, max: 250, step: 1 },
      { key: "volumeFilter", label: "量能濾網", type: "bool", default: false },
      { key: "volumeMaWindow", label: "量能均線天數", type: "number", default: 20, min: 5, max: 120, step: 1 },
      { key: "volumeMinRatio", label: "量能最小倍數", type: "number", default: 1.2, min: 0.5, max: 5, step: 0.1 },
      { key: "volatilitySqueezeFilter", label: "波動壓縮濾網", type: "bool", default: false },
      { key: "squeezeLookback", label: "壓縮回看天數", type: "number", default: 60, min: 10, max: 250, step: 1 },
      { key: "squeezePercentile", label: "壓縮分位數", type: "number", default: 0.5, min: 0.1, max: 0.9, step: 0.05 },
    ],
    rsi2: [
      { key: "rsiPeriod", label: "RSI天數", type: "number", default: 2, min: 2, max: 30, step: 1 },
      { key: "oversold", label: "超賣門檻", type: "number", default: 10, min: 1, max: 40, step: 1 },
      { key: "overbought", label: "超買門檻", type: "number", default: 70, min: 50, max: 99, step: 1 },
      { key: "trendMaWindow", label: "趨勢均線天數", type: "number", default: 200, min: 5, max: 250, step: 1 },
      { key: "maxHoldDays", label: "最長持有天數", type: "number", default: 10, min: 1, max: 60, step: 1 },
      { key: "slippagePoints", label: "滑價(點)", type: "number", default: 1, min: 0, max: 10, step: 0.5 },
    ],
    orb: [
      { key: "rangeStartMoy", label: "開盤區間起", type: "time", default: 525 },
      { key: "rangeEndMoy", label: "開盤區間迄", type: "time", default: 555 },
      { key: "sessionEndMoy", label: "收盤前強制平倉", type: "time", default: 805 },
      { key: "minRangePoints", label: "最小區間寬度(點)", type: "number", default: 5, min: 0, max: 100, step: 1 },
      { key: "slippagePoints", label: "滑價(點)", type: "number", default: 1, min: 0, max: 10, step: 0.5 },
      { key: "targetRMultiple", label: "停利R倍數(留空=不設，抱到收盤)", type: "nullable_number", default: null, min: 0.5, max: 10, step: 0.1 },
    ],
    liqsweep: [
      { key: "channelWindow", label: "流動性池回看天數", type: "number", default: 20, min: 5, max: 120, step: 1 },
      { key: "sweepBufferPoints", label: "刺穿緩衝(點)", type: "number", default: 10, min: 0, max: 100, step: 1 },
      { key: "stopBufferPoints", label: "停損緩衝(點)", type: "number", default: 5, min: 0, max: 100, step: 1 },
      { key: "targetRMultiple", label: "停利R倍數", type: "number", default: 2.0, min: 0.5, max: 10, step: 0.1 },
      { key: "maxHoldDays", label: "最長持有天數", type: "number", default: 10, min: 1, max: 60, step: 1 },
      { key: "slippagePoints", label: "滑價(點)", type: "number", default: 1, min: 0, max: 10, step: 0.5 },
    ],
    night: [
      { key: "windowStartMoy", label: "熱區開始", type: "time", default: 1260 },
      { key: "decisionMoy", label: "決策時間", type: "time", default: 1350 },
      { key: "windowEndMoy", label: "熱區結束", type: "time", default: 1425 },
      { key: "minMovePoints", label: "最小幅度(點)", type: "number", default: 20, min: 0, max: 200, step: 1 },
      { key: "slippagePoints", label: "滑價(點)", type: "number", default: 1, min: 0, max: 10, step: 0.5 },
      { key: "minDominantSideFraction", label: "多數方向門檻", type: "number", default: 0.90, min: 0.5, max: 1.0, step: 0.01 },
    ],
  };

  return {
    POINT_VALUE, TAX_RATE_PER_SIDE,
    ewmMean, shift1, rollingMean, rollingMax, rollingMin, rollingQuantile, pctChangeN,
    computeATR, computeRSI, fillPriceStop, limitFill, applyCosts,
    moyToHHMM, hhmmToMoy,
    parseMinuteBin, barAt, firstBarAtOrAfter,
    dailyFromRecords, buildDateIndexMap, alignMinuteBlocks,
    runOpeningRally, runLunchReversal, runTrendDay, runNightWindow,
    runSupportResistanceFade, runDonchian, runRsi2, runORB, runLiquiditySweep,
    computeVolRatioLag1, applyVolumeFilterFixed, applyTrendRegimeFilter, combine,
    STRATEGY_PARAMS,
  };
});
