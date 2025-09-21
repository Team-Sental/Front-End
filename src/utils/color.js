// Utility color helpers
// Returns a color based on occupancy percentage thresholds.
export function getBarColor(percent) {
  if (percent < 70) return "#16a34a"; // green
  if (percent < 90) return "#f59e0b"; // orange
  return "#ef4444"; // red
}
