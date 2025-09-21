// src/App.jsx
import React, { useEffect, useState } from "react";
import { VenueRow, AttendanceCard, ChartCard, RisksCard } from "./components";

// Main App component
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
          predicted:
            typeof item.predicted !== "undefined" ? item.predicted : null,
        }));

        if (mounted) {
          setData({
            attendance: d.attendance ?? 0,
            capacity:
              (d.capacity ??
                (Array.isArray(d.venues)
                  ? d.venues.reduce((s, v) => s + (v.capacity ?? 0), 0)
                  : 0)) ||
              1,
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

  if (!data)
    return <div style={{ color: "white", padding: 20 }}>Loading...</div>;

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
        overflowX: "hidden",
      }}
    >
      <div
        style={{
          width: "100%",
          color: "#e6eef8",
          margin: "0 auto",
          maxWidth: "1800px",
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
              {clock.toLocaleDateString()}
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
              {clock.toLocaleTimeString()}
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
        <div
          style={{ display: "grid", gridTemplateColumns: "2fr 1fr", gap: 12 }}
        >
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
