import React from "react";

export function RisksCard({ jsonRisk, autoRisks, lastUpdated }) {
  const combined = [...(autoRisks || [])];

  if (jsonRisk?.risk) {
    combined.push({
      severity: "info",
      type: jsonRisk.risk,
      zone: jsonRisk.zone || "",
      suggestion: jsonRisk.suggestion,
      detail: jsonRisk.detail || "",
    });
  }

  const severityStyles = {
    critical: {
      color: "#fecaca",
      bg: "rgba(239,68,68,0.12)",
      border: "#ef4444",
    },
    high: { color: "#fca5a5", bg: "rgba(239,68,68,0.08)", border: "#ef4444" },
    moderate: {
      color: "#fcd34d",
      bg: "rgba(250,204,21,0.08)",
      border: "#f59e0b",
    },
    watch: { color: "#93c5fd", bg: "rgba(59,130,246,0.08)", border: "#3b82f6" },
    info: { color: "#cbd5e1", bg: "#1e293b", border: "#334155" },
  };

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        minHeight: 0,
        overflow: "hidden",
      }}
    >
      <div
        style={{
          fontSize: 16,
          fontWeight: 700,
          color: "#fbbf24",
          marginBottom: 12,
        }}
      >
        Risks & Recommendations
      </div>
      <div style={{ fontSize: 12, color: "#64748b", marginBottom: 8 }}>
        Comprehensive rule-based assessment (capacity, trend, forecast
        deviation, surge prediction, load balance).
      </div>
      <div style={{ flex: 1, overflowY: "auto", paddingRight: 4 }}>
        {combined.length === 0 ? (
          <div style={{ color: "#94a3b8" }}>No risks detected</div>
        ) : (
          combined.map((r, i) => {
            const sev = severityStyles[r.severity] || severityStyles.info;
            return (
              <div
                key={i}
                style={{
                  background: sev.bg,
                  border: `1px solid ${sev.border}`,
                  borderRadius: 10,
                  padding: 12,
                  marginBottom: 10,
                  display: "flex",
                  flexDirection: "column",
                  gap: 6,
                }}
              >
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <span
                    style={{
                      background: sev.border,
                      color: "#0f172a",
                      fontSize: 10,
                      fontWeight: 700,
                      letterSpacing: 0.5,
                      padding: "4px 6px",
                      borderRadius: 4,
                      textTransform: "uppercase",
                    }}
                  >
                    {r.severity || "info"}
                  </span>
                  <span
                    style={{ fontWeight: 700, color: sev.color, fontSize: 14 }}
                  >
                    {r.type}
                    {r.zone ? ` — ${r.zone.replace(/_/g, " ")}` : ""}
                  </span>
                </div>
                {r.suggestion && (
                  <div
                    style={{ color: "#e2e8f0", fontSize: 13, lineHeight: 1.35 }}
                  >
                    {r.suggestion}
                  </div>
                )}
                {r.detail && (
                  <div style={{ color: "#94a3b8", fontSize: 12 }}>
                    {r.detail}
                  </div>
                )}
              </div>
            );
          })
        )}
      </div>
      <div style={{ marginTop: 8, fontSize: 11, color: "#64748b" }}>
        Last updated: {lastUpdated}
      </div>
    </div>
  );
}

export default RisksCard;
