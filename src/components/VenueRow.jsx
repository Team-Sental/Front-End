import React from "react";
import { getBarColor } from "./getBarColor";

function VenueRow({ venues }) {
  return (
    <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 12 }}>
      {venues.map((v) => {
        //Calculate from real time dummy data
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

export default VenueRow;