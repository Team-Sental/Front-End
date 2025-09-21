import React from "react";

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

export default RisksCard;