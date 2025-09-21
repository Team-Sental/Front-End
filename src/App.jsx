// src/App.jsx
import React, { useEffect, useState, useRef } from "react";
import { VenueRow, AttendanceCard, ChartCard, RisksCard } from "./components";

// Main App component
export default function App() {
  const [data, setData] = useState(null); // transformed API/output snapshot
  const [error, setError] = useState(null);
  const [debug, setDebug] = useState({
    attempts: 0,
    lastUrl: null,
    lastStatus: null,
  });
  const [selectedVenue, setSelectedVenue] = useState(null);
  const [lastUpdated, setLastUpdated] = useState(null);
  const [actualHistory, setActualHistory] = useState({}); // { venueName: [densities over steps] }
  const runningRef = useRef(true);
  const timerRef = useRef(null);

  // Base API endpoint (could later be moved to env var)
  const API_BASE =
    "https://jhtl8la4ik.execute-api.ap-southeast-1.amazonaws.com/dev";

  // Transform the output.json structure into internal state shape
  function transformOutput(json) {
    // Normalize timestamp (replace space with T for Safari/edge cases)
    const ts =
      typeof json.timestamp === "string"
        ? json.timestamp.replace(" ", "T")
        : json.timestamp;
    const baseTimestamp = new Date(ts);
    const locations = Array.isArray(json.locations) ? json.locations : [];

    const venues = locations.map((loc) => {
      const row = json.rows?.[loc] || {};
      return {
        name: loc,
        current: row.density ?? 0,
        capacity: row.max_crowd ?? 1,
      };
    });

    const attendance = locations.reduce(
      (sum, loc) => sum + (json.rows?.[loc]?.density ?? 0),
      0
    );
    const capacity = locations.reduce(
      (sum, loc) => sum + (json.rows?.[loc]?.max_crowd ?? 0),
      0
    );

    const entryGateRow = json.rows?.entry_gate;
    const weather = entryGateRow
      ? {
          temp: entryGateRow.temperature,
          cond: entryGateRow.weather,
          wind: entryGateRow.windspeed,
        }
      : null;

    const predictions = {};
    locations.forEach((loc) => {
      predictions[loc] = json.predictions?.[loc]?.prediction?.mean || [];
    });

    if (!json.timestamp || !locations.length) {
      throw new Error(
        `Malformed payload: timestamp=${json.timestamp} locations=${locations.length}`
      );
    }

    return {
      baseTimestamp,
      stepIndex: json.stepIndex,
      nextStepIndex: json.nextStepIndex,
      done: json.done,
      totalSteps: json.total_steps,
      locations,
      venues,
      attendance,
      capacity: capacity || 1,
      weather,
      predictions, // { loc: [predicted values] }
    };
  }

  // Polling loop for /predict?stepIndex=...
  useEffect(() => {
    let mounted = true;

    async function poll(stepIndex = 0, isRetry = false) {
      if (!runningRef.current) return;
      try {
        const url = `${API_BASE}/predict?stepIndex=${stepIndex}`;
        setDebug((d) => ({ ...d, attempts: d.attempts + 1, lastUrl: url }));
        const res = await fetch(url, { cache: "no-store" });
        if (!res.ok) {
          setDebug((d) => ({ ...d, lastStatus: res.status }));
          throw new Error(`Network error: ${res.status}`);
        }
        const json = await res.json();
        setDebug((d) => ({ ...d, lastStatus: 200 }));
        if (!mounted) return;
        const transformed = transformOutput(json);
        console.debug("[poll] raw json keys:", Object.keys(json));
        console.debug("[poll] transformed:", {
          stepIndex: transformed.stepIndex,
          venues: transformed.venues.length,
          locations: transformed.locations.length,
        });
        setError(null);
        setData(transformed);
        console.debug("[poll] state set: data not null");
        // Append actual densities for this step
        setActualHistory((prev) => {
          const updated = { ...prev };
          transformed.venues.forEach((v) => {
            const arr = updated[v.name] || [];
            if (arr.length === transformed.stepIndex) {
              updated[v.name] = [...arr, v.current];
            } else if (arr.length < transformed.stepIndex + 1) {
              // Fill any gaps (shouldn't normally happen)
              const filler = Array(transformed.stepIndex - arr.length).fill(
                null
              );
              updated[v.name] = [...arr, ...filler, v.current];
            }
          });
          return updated;
        });
        setLastUpdated(new Date().toLocaleString());
        setSelectedVenue((prev) =>
          transformed.venues.some((v) => v.name === prev) ? prev : null
        );

        if (transformed.done) {
          runningRef.current = false;
          return;
        }
        // Schedule next poll at 3s
        timerRef.current = setTimeout(
          () => poll(transformed.nextStepIndex ?? transformed.stepIndex + 1),
          3000
        );
      } catch (e) {
        console.error("Polling failed", e);
        if (!isRetry && stepIndex === 0) {
          // Single fallback attempt to local static file so user sees data if API blocked
          try {
            const localRes = await fetch("/output.json", { cache: "no-store" });
            if (localRes.ok) {
              const localJson = await localRes.json();
              const transformedLocal = transformOutput(localJson);
              console.debug("[fallback] using local snapshot");
              if (!mounted) return;
              setData(transformedLocal);
              setError("Live API failed, showing static snapshot.");
              setDebug((d) => ({ ...d, lastStatus: "fallback-local" }));
              return; // stop polling until manual refresh
            }
          } catch (inner) {
            console.warn("Local fallback also failed", inner);
          }
        }
        setError(e.message || "Unknown polling error");
        // Retry after a delay
        timerRef.current = setTimeout(() => poll(stepIndex, true), 5000);
      }
    }

    poll(0);
    // Timed fallback: if after 6000ms still no data and no error, try local snapshot
    const safety = setTimeout(async () => {
      if (!data && !error) {
        try {
          const r = await fetch("/output.json", { cache: "no-store" });
          if (r.ok) {
            const j = await r.json();
            const t = transformOutput(j);
            if (runningRef.current) {
              setData(t);
              setError("Using static snapshot (API slow)");
              setDebug((d) => ({ ...d, lastStatus: "timeout-fallback" }));
            }
          }
        } catch {}
      }
    }, 6000);
    return () => {
      mounted = false;
      runningRef.current = false;
      if (timerRef.current) clearTimeout(timerRef.current);
      clearTimeout(safety);
    };
  }, []);

  if (!data) {
    return (
      <div
        style={{ color: "white", padding: 24, fontFamily: "Inter, sans-serif" }}
      >
        <div style={{ fontSize: 18, fontWeight: 600, marginBottom: 8 }}>
          Loading live data…
        </div>
        <div
          style={{
            fontSize: 14,
            color: error ? "#f87171" : "#94a3b8",
            marginBottom: 6,
          }}
        >
          {error ? `Last error: ${error}` : "Contacting prediction API…"}
        </div>
        <div style={{ fontSize: 12, color: "#64748b" }}>
          Attempts: {debug.attempts} | Last status: {String(debug.lastStatus)}
          {debug.lastUrl && (
            <div style={{ marginTop: 4, wordBreak: "break-all" }}>
              Last URL: {debug.lastUrl}
            </div>
          )}
        </div>
      </div>
    );
  }

  // Advanced rule-based risk detection
  const autoRisks = [];
  const nowStep = data.stepIndex;
  const horizon = 6; // look ahead 6 prediction points (~30m)

  data.venues.forEach((v) => {
    const pct = v.capacity ? v.current / v.capacity : 0;
    const predicted = data.predictions[v.name] || [];
    const upcoming = predicted.slice(nowStep + 1, nowStep + 1 + horizon);
    const maxUpcoming = upcoming.length ? Math.max(...upcoming) : null;
    const currentPredicted = predicted[nowStep] ?? null;
    const recentActual = actualHistory[v.name] || [];
    const trendWindow = recentActual.slice(-4);
    const trendDirection =
      trendWindow.length >= 2
        ? trendWindow[trendWindow.length - 1] - trendWindow[0]
        : 0;

    function pushRisk({ severity, type, suggestion, detail }) {
      autoRisks.push({
        severity,
        type,
        zone: v.name,
        suggestion,
        detail,
      });
    }

    if (pct >= 0.98) {
      pushRisk({
        severity: "critical",
        type: "Overcapacity",
        suggestion: `Immediate action required at ${v.name}. Halt inflow / reroute crowds.`,
        detail: `Utilization ${(pct * 100).toFixed(
          1
        )}% exceeds safe threshold (98%).`,
      });
    } else if (pct >= 0.9) {
      pushRisk({
        severity: "high",
        type: "Near Capacity",
        suggestion: `Deploy staff to ${v.name} for crowd regulation and prepare overflow paths.`,
        detail: `Current load ${(pct * 100).toFixed(1)}%.`,
      });
    } else if (pct >= 0.75) {
      pushRisk({
        severity: "moderate",
        type: "Rising Load",
        suggestion: `Increase monitoring cadence; consider soft steering signage for ${v.name}.`,
        detail: `Load ${(pct * 100).toFixed(1)}%.`,
      });
    }

    if (trendDirection > v.capacity * 0.1) {
      pushRisk({
        severity: pct > 0.85 ? "high" : "moderate",
        type: "Rapid Inflow",
        suggestion: `Inflow accelerating at ${v.name}; pre-empt congestion measures (staff reposition / signage).`,
        detail: `Δ last ${
          trendWindow.length
        } points = +${trendDirection.toLocaleString()}.`,
      });
    }

    if (currentPredicted && v.current > currentPredicted * 1.15) {
      pushRisk({
        severity: "moderate",
        type: "Above Forecast",
        suggestion: `Actual attendance surpasses forecast at ${v.name}; update predictive model inputs or investigate anomaly.`,
        detail: `Actual ${v.current} vs forecast ${Math.round(
          currentPredicted
        )} (+${((v.current / currentPredicted - 1) * 100).toFixed(1)}%).`,
      });
    }

    if (maxUpcoming && maxUpcoming / (v.capacity || 1) > 0.95) {
      pushRisk({
        severity: "watch",
        type: "Forecast Surge",
        suggestion: `Prepare mitigation—forecast indicates surge at ${
          v.name
        } within next ${(upcoming.indexOf(maxUpcoming) + 1) * 5} minutes.`,
        detail: `Peak forecast ${Math.round(maxUpcoming)} (${(
          (maxUpcoming / (v.capacity || 1)) *
          100
        ).toFixed(1)}% capacity).`,
      });
    }
  });

  // Imbalance detection (one venue much higher than median)
  const loads = data.venues.map((v) => ({
    name: v.name,
    pct: v.capacity ? v.current / v.capacity : 0,
  }));
  if (loads.length) {
    const medianPct = [...loads].sort((a, b) => a.pct - b.pct)[
      Math.floor(loads.length / 2)
    ].pct;
    loads.forEach((l) => {
      if (l.pct > medianPct * 1.5 && l.pct > 0.6) {
        autoRisks.push({
          severity: l.pct > 0.85 ? "high" : "moderate",
          type: "Load Imbalance",
          zone: l.name,
          suggestion: `Redistribute flow: ${l.name.replace(
            /_/g,
            " "
          )} disproportionately loaded vs median.`,
          detail: `Load ${(l.pct * 100).toFixed(1)}% vs median ${(
            medianPct * 100
          ).toFixed(1)}%.`,
        });
      }
    });
  }

  autoRisks.sort((a, b) => {
    const order = { critical: 4, high: 3, moderate: 2, watch: 1 };
    return (order[b.severity] || 0) - (order[a.severity] || 0);
  });

  return (
    <div
      style={{
        height: "100vh",
        display: "flex",
        flexDirection: "column",
        background: "#0b1220",
        fontFamily: "Inter, Roboto, sans-serif",
        overflow: "hidden",
      }}
    >
      <div
        style={{
          flex: 1,
          display: "flex",
          flexDirection: "column",
          padding: 20,
          width: "100%",
          color: "#e6eef8",
          minHeight: 0,
        }}
      >
        {/* Top row: Date | Time | Weather | Wind | Attendance (spans 2) */}
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(6, minmax(0,1fr))",
            gap: 12,
            marginBottom: 16,
          }}
        >
          {/* Date Card */}
          <div style={{ background: "#111827", padding: 16, borderRadius: 12 }}>
            <div
              style={{
                fontSize: 13,
                letterSpacing: 0.5,
                color: "#93c5fd",
                textTransform: "uppercase",
              }}
            >
              Date
            </div>
            <div style={{ fontSize: 24, fontWeight: 700, lineHeight: 1.1 }}>
              {data.baseTimestamp.toLocaleDateString()}
            </div>
          </div>

          {/* Time Card */}
          <div style={{ background: "#111827", padding: 16, borderRadius: 12 }}>
            <div
              style={{
                fontSize: 13,
                letterSpacing: 0.5,
                color: "#93c5fd",
                textTransform: "uppercase",
              }}
            >
              Time
            </div>
            <div style={{ fontSize: 28, fontWeight: 700, lineHeight: 1.1 }}>
              {data.baseTimestamp.toLocaleTimeString([], {
                hour: "2-digit",
                minute: "2-digit",
                second: "2-digit",
              })}
            </div>
          </div>

          {/* Weather Card */}
          <div style={{ background: "#111827", padding: 16, borderRadius: 12 }}>
            <div
              style={{
                fontSize: 13,
                letterSpacing: 0.5,
                color: "#93c5fd",
                textTransform: "uppercase",
              }}
            >
              Weather
            </div>
            <div style={{ fontSize: 32, fontWeight: 700 }}>
              {data.weather ? `${Math.round(data.weather.temp)}°C` : "—"}
            </div>
            <div style={{ fontSize: 14, color: "#9ca3af", fontWeight: 500 }}>
              {data.weather ? data.weather.cond : "Placeholder"}
            </div>
          </div>

          {/* Wind Speed Card */}
          <div style={{ background: "#111827", padding: 16, borderRadius: 12 }}>
            <div
              style={{
                fontSize: 13,
                letterSpacing: 0.5,
                color: "#93c5fd",
                textTransform: "uppercase",
              }}
            >
              Wind Speed
            </div>
            <div style={{ fontSize: 30, fontWeight: 700 }}>
              {data.weather ? `${data.weather.wind} km/h` : "—"}
            </div>
            <div style={{ fontSize: 13, color: "#9ca3af" }}>
              {data.weather ? "Current" : "Placeholder"}
            </div>
          </div>

          {/* Attendance Card spanning 2 columns */}
          <div style={{ gridColumn: "span 2" }}>
            <AttendanceCard
              attendance={data.attendance}
              capacity={data.capacity}
            />
          </div>
        </div>

        {/* Combined main content: Left (venues + chart) | Right (risks) */}
        <div
          style={{
            flex: 1,
            minHeight: 0,
            display: "grid",
            gridTemplateColumns: "2fr 1fr",
            gap: 16,
          }}
        >
          {/* Left combined card */}
          <div
            style={{
              background: "#111827",
              borderRadius: 12,
              padding: 16,
              display: "flex",
              flexDirection: "column",
              minHeight: 0,
            }}
          >
            <div
              style={{
                display: "flex",
                alignItems: "center",
                marginBottom: 12,
              }}
            >
              <div
                style={{
                  fontSize: 18,
                  fontWeight: 800,
                  color: "#fbbf24",
                  flex: 1,
                }}
              >
                Venue Occupancy
              </div>
              {selectedVenue && (
                <button
                  onClick={() => setSelectedVenue(null)}
                  style={{
                    background: "#1e293b",
                    border: "1px solid #334155",
                    color: "#e2e8f0",
                    fontSize: 12,
                    padding: "6px 10px",
                    borderRadius: 6,
                    cursor: "pointer",
                  }}
                >
                  Clear Selection
                </button>
              )}
            </div>
            <div style={{ marginBottom: 16 }}>
              <VenueRow
                venues={data.venues}
                selected={selectedVenue}
                onSelect={(name) =>
                  setSelectedVenue((prev) => (prev === name ? null : name))
                }
              />
            </div>
            <div
              style={{
                flex: 1,
                minHeight: 0,
                display: "flex",
                flexDirection: "column",
              }}
            >
              {selectedVenue ? (
                (() => {
                  const predicted = data.predictions[selectedVenue] || [];
                  const actualArr = actualHistory[selectedVenue] || [];
                  const graphData = predicted.map((p, idx) => {
                    const t = new Date(
                      data.baseTimestamp.getTime() + idx * 5 * 60 * 1000
                    );
                    const label = t.toLocaleTimeString([], {
                      hour: "2-digit",
                      minute: "2-digit",
                    });
                    return {
                      time: label,
                      predicted: p,
                      actual: idx < actualArr.length ? actualArr[idx] : null,
                    };
                  });
                  return (
                    <ChartCard
                      graphData={graphData}
                      venueName={selectedVenue}
                    />
                  );
                })()
              ) : (
                <div
                  style={{
                    background: "#0f172a",
                    border: "1px dashed #334155",
                    borderRadius: 10,
                    padding: 24,
                    color: "#64748b",
                    fontSize: 14,
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    flex: 1,
                  }}
                >
                  Select a venue to view its trend
                </div>
              )}
            </div>
          </div>
          {/* Right risks card with internal scroll */}
          <div
            style={{
              background: "#111827",
              borderRadius: 12,
              padding: 16,
              display: "flex",
              flexDirection: "column",
              minHeight: 0,
              overflow: "hidden",
            }}
          >
            <RisksCard
              jsonRisk={data.risksSource}
              autoRisks={autoRisks}
              lastUpdated={lastUpdated}
            />
          </div>
        </div>
      </div>
    </div>
  );
}
