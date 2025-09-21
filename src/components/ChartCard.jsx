import React from "react";
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

// Custom tooltip component matching dark theme
function CustomTooltip({ active, payload, label }) {
  if (!active || !payload || payload.length === 0) return null;
  const actual = payload.find((p) => p.dataKey === "actual");
  const predicted = payload.find((p) => p.dataKey === "predicted");
  const delta = actual && predicted ? actual.value - predicted.value : null;
  const deltaColor =
    delta === null ? "#94a3b8" : delta > 0 ? "#ef4444" : "#16a34a";

  return (
    <div
      style={{
        background: "#0f172a",
        border: "1px solid #1e293b",
        boxShadow: "0 4px 14px rgba(0,0,0,0.4)",
        borderRadius: 10,
        padding: "10px 12px 12px",
        minWidth: 160,
        fontSize: 12,
        color: "#e2e8f0",
        backdropFilter: "blur(4px)",
      }}
    >
      <div
        style={{
          fontSize: 11,
          letterSpacing: 0.5,
          color: "#93c5fd",
          marginBottom: 4,
        }}
      >
        {label}
      </div>
      {actual && (
        <div style={{ display: "flex", alignItems: "center", marginBottom: 4 }}>
          <span
            style={{
              display: "inline-block",
              width: 8,
              height: 8,
              borderRadius: 4,
              background: actual.color || "#4A90E2",
              marginRight: 6,
            }}
          />
          <span style={{ color: "#cbd5e1" }}>Actual:</span>
          <span style={{ marginLeft: "auto", fontWeight: 600 }}>
            {actual.value?.toLocaleString?.() ?? actual.value}
          </span>
        </div>
      )}
      {predicted && (
        <div style={{ display: "flex", alignItems: "center", marginBottom: 4 }}>
          <span
            style={{
              display: "inline-block",
              width: 8,
              height: 8,
              borderRadius: 4,
              background: predicted.color || "#F59E0B",
              marginRight: 6,
            }}
          />
          <span style={{ color: "#cbd5e1" }}>Predicted:</span>
          <span style={{ marginLeft: "auto", fontWeight: 600 }}>
            {predicted.value?.toLocaleString?.() ?? predicted.value}
          </span>
        </div>
      )}
      {delta !== null && (
        <div style={{ display: "flex", alignItems: "center", marginTop: 2 }}>
          <span style={{ color: "#94a3b8" }}>Delta:</span>
          <span
            style={{ marginLeft: "auto", fontWeight: 600, color: deltaColor }}
          >
            {delta > 0 ? "+" : ""}
            {delta.toLocaleString()}
          </span>
        </div>
      )}
    </div>
  );
}

export function ChartCard({ graphData, venueName }) {
  return (
    <div
      style={{
        background: "#111827",
        padding: 16,
        borderRadius: 10,
        color: "white",
      }}
    >
      <div style={{ fontSize: 14, color: "#cbd5e1", marginBottom: 6 }}>
        {venueName
          ? venueName.replace(/_/g, " ") + " Trend"
          : "Attendance Trend"}
      </div>
      <div style={{ height: 300 }}>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={graphData}>
            <CartesianGrid stroke="#2d3748" />
            <XAxis dataKey="time" stroke="#94a3b8" />
            <YAxis stroke="#94a3b8" />
            <Tooltip
              content={<CustomTooltip />}
              cursor={{ stroke: "#334155", strokeWidth: 1 }}
              wrapperStyle={{ outline: "none" }}
            />
            <Legend />
            <Line
              type="monotone"
              dataKey="actual"
              name="Actual Data"
              stroke="#4A90E2"
              dot={{ r: 4 }}
              activeDot={{ r: 6 }}
              connectNulls={false}
              isAnimationActive={false}
            />
            <Line
              type="monotone"
              dataKey="predicted"
              name="Predicted Data"
              stroke="#F59E0B"
              strokeDasharray="5 5"
              dot={{ r: 4 }}
              connectNulls={false}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
      {(!graphData || graphData.length === 0) && (
        <div style={{ fontSize: 12, color: "#64748b", marginTop: 8 }}>
          No data for this venue.
        </div>
      )}
    </div>
  );
}

export default ChartCard;
