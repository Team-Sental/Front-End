// src/App.jsx
import React, { useEffect, useState } from "react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Legend,
} from "recharts";

// Color helper based on percentage
function getBarColor(percent) {
  if (percent < 70) return "#16a34a"; // green
  if (percent < 90) return "#f59e0b"; // orange
  return "#ef4444"; // red
}

// Venue occupancy row
function VenueRow({ venues }) {
  return (
    <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 12 }}>
      {venues.map((v) => {
        const pct = v.capacity ? Math.min(100, Math.round((v.current / v.capacity) * 100)) : 0;
        return (
          <div
            key={v.name}
            style={{
              background: "#1f2937",
              padding: 12,
              borderRadius: 10,
              color: "white",
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
            }}
          >
            <div style={{ fontWeight: 700, marginBottom: 6, textTransform: "capitalize" }}>
              {v.name.replace(/_/g, " ")}
            </div>
            <div style={{ fontSize: 14, color: "#cbd5e1", marginBottom: 8 }}>
              {v.current.toLocaleString()} / {v.capacity.toLocaleString()}
            </div>
            <div style={{ width: "100%", background: "#374151", height: 10, borderRadius: 6 }}>
              <div
                style={{
                  width: `${pct}%`,
                  height: "100%",
                  borderRadius: 6,
                  background: getBarColor(pct),
                }}
              />
            </div>
            <div style={{ fontSize: 12, color: "#94a3b8", marginTop: 8 }}>{pct}%</div>
          </div>
        );
      })}
    </div>
  );
}

// Live attendance card
function AttendanceCard({ attendance, capacity }) {
  
  const percent = capacity ? Math.min(100, Math.round((attendance / capacity) * 100)) : 0;
  const color = getBarColor(percent);

  return (
    <div style={{ background: "#111827", padding: 16, borderRadius: 10, color: "white" }}>
      <div style={{ fontSize: 14, color: "#cbd5e1" }}>Live Attendance</div>
      <div style={{ fontSize: 28, fontWeight: 700 }}>{attendance.toLocaleString()}</div>
      <div style={{ fontSize: 12, color: "#94a3b8", marginBottom: 8 }}>
        of {capacity.toLocaleString()} total capacity
      </div>
      <div style={{ width: "100%", background: "#374151", height: 12, borderRadius: 8 }}>
        <div style={{ width: `${percent}%`, height: "100%", borderRadius: 8, background: color }} />
      </div>
      <div style={{ fontSize: 12, color: "#9ca3af", marginTop: 6 }}>{percent}%</div>
    </div>
  );
}

// Attendance trend chart
function ChartCard({ graphData }) {
  return (
    <div style={{ background: "#111827", padding: 16, borderRadius: 10, color: "white" }}>
      <div style={{ fontSize: 14, color: "#cbd5e1", marginBottom: 6 }}>Attendance Trend</div>
      <div style={{ height: 300 }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={graphData}>
            <CartesianGrid stroke="#2d3748" />
            <XAxis dataKey="time" stroke="#94a3b8" />
            <YAxis stroke="#94a3b8" />
            <Tooltip />
            <Legend />
            <Line
              type="monotone"
              dataKey="actual"
              name="Actual Data"
              stroke="#4A90E2"
              dot={{ r: 4 }}
              activeDot={{ r: 6 }}
              connectNulls={false}
            />
            <Line
              type="monotone"
              dataKey="predicted"
              name="Predicted Data"
              stroke="#F59E0B"
              strokeDasharray="5 5"
              dot={{ r: 4 }}
              connectNulls={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

// Risks & suggestions
function RisksCard({ jsonRisk, autoRisks, lastUpdated }) {
  const combined = [...(autoRisks || [])];

  if (jsonRisk?.risk) {
    combined.push({
      type: "Source",
      zone: "",
      suggestion: jsonRisk.suggestion,
      label: jsonRisk.risk,
    });
  }

  return (
    <div style={{ background: "#111827", padding: 16, borderRadius: 10, color: "white" }}>
      <div style={{ fontSize: 14, color: "#cbd5e1", marginBottom: 8 }}>Risks & Suggestions</div>
      {combined.length === 0 ? (
        <div style={{ color: "#94a3b8" }}>No risks detected</div>
      ) : (
        combined.map((r, i) => (
          <div
            key={i}
            style={{ background: "#0f1720", padding: 10, borderRadius: 8, marginBottom: 8 }}
          >
            <div
              style={{
                fontWeight: 700,
                color: r.type === "Overcrowding" ? "#ef4444" : "#f59e0b",
              }}
            >
              {r.type}
              {r.zone ? ` — ${r.zone}` : ""}
            </div>
            <div style={{ color: "#93c5fd", marginTop: 4 }}>{r.suggestion || r.label}</div>
          </div>
        ))
      )}
      <div style={{ marginTop: 8, fontSize: 12, color: "#94a3b8" }}>
        Last updated: {lastUpdated}
      </div>
    </div>
  );
}

// Main App
export default function App() {
  const [data, setData] = useState(null);
  const [clock, setClock] = useState(new Date());
  const [lastUpdated, setLastUpdated] = useState(null);

  // Update clock every second
  useEffect(() => {
    const t = setInterval(() => setClock(new Date()), 1000);
    return () => clearInterval(t);
  }, []);

  // Fetch data.json every 5 minutes
  useEffect(() => {
    let mounted = true;

    async function fetchData() {
      try {
        const res = await fetch("/data.json", { cache: "no-store" });
        if (!res.ok) throw new Error("Network response was not ok");
        const d = await res.json();

        const normalizedGraph = (d.graphData || []).map((item) => ({
          time: item.time ?? item.day ?? "",
          actual: typeof item.actual !== "undefined" ? item.actual : null,
          predicted: typeof item.predicted !== "undefined" ? item.predicted : null,
        }));

        if (mounted) {
          setData({
            attendance: d.attendance ?? 0,
            capacity: (d.capacity ?? (Array.isArray(d.venues) ? d.venues.reduce((s, v) => s + (v.capacity ?? 0), 0) : 0)) || 1,
            venues: Array.isArray(d.venues) ? d.venues : [],
            graphData: normalizedGraph,
            risksSource: d.risks ?? null,
            weather: d.weather ?? null,
          });
          setLastUpdated(new Date().toLocaleString());
        }
      } catch (err) {
        console.error("Failed to load data.json:", err);
      }
    }

    fetchData();
    const interval = setInterval(fetchData, 300_000); // 5 minutes
    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, []);

  if (!data) return <div style={{ color: "white", padding: 20 }}>Loading...</div>;

  // Auto-detect risks
  const autoRisks = [];
  data.venues.forEach((v) => {
    const pct = v.capacity ? v.current / v.capacity : 0;
    if (pct >= 0.95) {
      autoRisks.push({
        type: "Overcrowding",
        zone: v.name,
        suggestion: `Critical overcrowding at ${v.name}: consider immediate redirection or capacity control.`,
      });
    } else if (pct >= 0.85) {
      autoRisks.push({
        type: "High load",
        zone: v.name,
        suggestion: `High load at ${v.name}: consider opening additional gates or delaying entry.`,
      });
    } else if (pct >= 0.7) {
      autoRisks.push({
        type: "Potential Bottleneck",
        zone: v.name,
        suggestion: `Monitor ${v.name}; consider crowd steering to adjacent areas.`,
      });
    }
  });

  return (
    <div
      style={{
        minHeight: "100vh",
        background: "#0b1220",
        padding: 20,
        fontFamily: "Inter, Roboto, sans-serif",
      }}
    >
      <div style={{ width: "100vw", color: "#e6eef8" }}>
        {/* Top row */}
        <div style={{ display: "flex", gap: 12, marginBottom: 16 }}>
          <div style={{ flex: 1, background: "#111827", padding: 12, borderRadius: 10 }}>
            <div style={{ fontSize: 12, color: "#93c5fd" }}>Date & Time</div>
            <div style={{ fontSize: 18, fontWeight: 700 }}>{clock.toLocaleDateString()}</div>
            <div style={{ fontSize: 16 }}>{clock.toLocaleTimeString()}</div>
          </div>

          <div style={{ width: 320 }}>
            <AttendanceCard attendance={data.attendance} capacity={data.capacity} />
          </div>

          <div style={{ width: 240, background: "#111827", padding: 12, borderRadius: 10 }}>
            <div style={{ fontSize: 12, color: "#93c5fd" }}>Weather</div>
            <div style={{ fontSize: 18, fontWeight: 700 }}>
              {data.weather ? `${data.weather.temp}°C` : "—"}
            </div>
            <div style={{ fontSize: 13, color: "#9ca3af" }}>
              {data.weather
                ? `${data.weather.cond} · Wind ${data.weather.wind} km/h`
                : "Placeholder"}
            </div>
          </div>
        </div>

        {/* Middle: Venue occupancy */}
        <div style={{ marginBottom: 16 }}>
          <div
            style={{
              fontSize: 18,
              color: "#fbbf24",
              fontWeight: 800,
              marginBottom: 12,
            }}
          >
            Venue Occupancy
          </div>
          <VenueRow venues={data.venues} />
        </div>

        {/* Bottom row: Chart + Risks */}
        <div style={{ display: "grid", gridTemplateColumns: "2fr 1fr", gap: 12 }}>
          <ChartCard graphData={data.graphData} />
          <RisksCard
            jsonRisk={data.risksSource}
            autoRisks={autoRisks}
            lastUpdated={lastUpdated}
          />
        </div>
      </div>
    </div>
  );
}
