import { useEffect, useMemo, useState } from "react";
import "./App.css";

const API_BASE = "http://127.0.0.1:8000";

function App() {
  const [dates, setDates] = useState([]);
  const [selectedDate, setSelectedDate] = useState("2020-05-25");
  const [depths, setDepths] = useState([]);
  const [selectedDepth, setSelectedDepth] = useState(20);
  const [selectedMap, setSelectedMap] = useState(null);

  const [loadingPrediction, setLoadingPrediction] = useState(false);
  const [loadingMap, setLoadingMap] = useState(false);
  const [error, setError] = useState("");

  const selectedDepthData = useMemo(
    () => depths.find((item) => item.depth_m === selectedDepth),
    [depths, selectedDepth]
  );

  useEffect(() => {
    loadDates();
  }, []);

  async function loadDates() {
    try {
      const response = await fetch(`${API_BASE}/dates`);

      if (!response.ok) {
        throw new Error("Failed to load available dates.");
      }

      const data = await response.json();

      setDates(data.dates || []);

      if (data.dates?.length > 0 && !data.dates.includes(selectedDate)) {
        setSelectedDate(data.dates[0]);
      }
    } catch (err) {
      setError(err.message);
    }
  }

  async function runPrediction() {
    setError("");
    setLoadingPrediction(true);

    try {
      const response = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          date: selectedDate,
        }),
      });

      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.detail || "Prediction failed.");
      }

      const data = await response.json();

      setDepths(data.depths || []);

      if (data.depths?.length > 0) {
        setSelectedDepth(data.depths[3]?.depth_m ?? data.depths[0].depth_m);
      }

      await loadMap(
        selectedDate,
        data.depths?.[3]?.depth_m ?? data.depths?.[0]?.depth_m
      );
    } catch (err) {
      setError(err.message);
    } finally {
      setLoadingPrediction(false);
    }
  }

  async function loadMap(date, depth) {
    if (depth === undefined) return;

    setError("");
    setLoadingMap(true);

    try {
      const response = await fetch(
        `${API_BASE}/map?date=${encodeURIComponent(
          date
        )}&depth=${encodeURIComponent(depth)}`
      );

      if (!response.ok) {
        const data = await response.json();
        throw new Error(data.detail?.message || "Failed to load map.");
      }

      const data = await response.json();

      setSelectedMap(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoadingMap(false);
    }
  }

  function handleDepthChange(event) {
    const depth = Number(event.target.value);

    setSelectedDepth(depth);

    if (dates.length > 0) {
      loadMap(selectedDate, depth);
    }
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">≈</div>

          <div>
            <h1>OceanEmbed</h1>
            <p>Subsurface Ocean Temperature Reconstruction</p>
          </div>
        </div>

        <div className="status-badge">
          <span className="status-dot" />
          MODEL ONLINE
        </div>
      </header>

      <main className="dashboard">
        <section className="hero">
          <div>
            <span className="eyebrow">BAY OF BENGAL • 2020</span>

            <h2>
              See beneath
              <br />
              the surface.
            </h2>

            <p>
              Reconstructing subsurface ocean temperature from sparse
              surface observations using SST, SSH and SSS.
            </p>
          </div>

          <div className="hero-grid">
            <div>
              <strong>76 × 76</strong>
              <span>Spatial grid</span>
            </div>

            <div>
              <strong>14</strong>
              <span>Depth levels</span>
            </div>

            <div>
              <strong>700 m</strong>
              <span>Maximum depth</span>
            </div>
          </div>
        </section>

        <section className="control-panel">
          <div className="control-group">
            <label>Date</label>

            <select
              value={selectedDate}
              onChange={(event) => setSelectedDate(event.target.value)}
            >
              {dates.length === 0 ? (
                <option>Loading...</option>
              ) : (
                dates.map((date) => (
                  <option key={date} value={date}>
                    {date}
                  </option>
                ))
              )}
            </select>
          </div>

          <button
            className="predict-button"
            onClick={runPrediction}
            disabled={loadingPrediction || dates.length === 0}
          >
            {loadingPrediction ? "Reconstructing..." : "Reconstruct"}
            <span>→</span>
          </button>
        </section>

        {error && (
          <div className="error-box">
            <strong>Something went wrong</strong>
            <span>{error}</span>
          </div>
        )}

        <section className="workspace">
          <div className="map-card">
            <div className="card-header">
              <div>
                <span className="eyebrow">TEMPERATURE FIELD</span>

                <h3>
                  {selectedMap
                    ? `${selectedMap.depth_m} m`
                    : `${selectedDepth} m`}
                </h3>
              </div>

              <div className="map-date">
                {selectedMap?.date || selectedDate}
              </div>
            </div>

            <div className="map-container">
              {loadingMap ? (
                <div className="map-loading">
                  <div className="spinner" />
                  <span>Loading temperature field...</span>
                </div>
              ) : selectedMap ? (
                <TemperatureMap
  map={selectedMap.temperature_map}
  oceanMask={selectedMap.ocean_mask}
/>
              ) : (
                <div className="map-empty">
                  <span>≈</span>
                  <p>Select a date and reconstruct the field.</p>
                </div>
              )}
            </div>

            <div className="map-footer">
              <span>5.5°N — 24.25°N</span>
              <span>80.5°E — 99.25°E</span>
            </div>
          </div>

          <aside className="side-panel">
            <div className="depth-card">
              <div className="card-header compact">
                <div>
                  <span className="eyebrow">DEPTH</span>
                  <h3>{selectedDepth} m</h3>
                </div>
              </div>

              <input
                type="range"
                min="0"
                max={Math.max(depths.length - 1, 0)}
                step="1"
                value={Math.max(
                  depths.findIndex(
                    (item) => item.depth_m === selectedDepth
                  ),
                  0
                )}
                onChange={(event) => {
                  const index = Number(event.target.value);

                  if (depths[index]) {
                    handleDepthChange({
                      target: {
                        value: depths[index].depth_m,
                      },
                    });
                  }
                }}
                disabled={depths.length === 0}
              />

              <div className="depth-scale">
                <span>Surface</span>
                <span>700 m</span>
              </div>

              <div className="depth-pills">
                {depths.map((item) => (
                  <button
                    key={item.depth_m}
                    className={
                      item.depth_m === selectedDepth
                        ? "depth-pill active"
                        : "depth-pill"
                    }
                    onClick={() =>
                      handleDepthChange({
                        target: {
                          value: item.depth_m,
                        },
                      })
                    }
                  >
                    {item.depth_m}m
                  </button>
                ))}
              </div>
            </div>

            <div className="stats-card">
              <span className="eyebrow">FIELD STATISTICS</span>

              <div className="stat-row">
                <span>Mean</span>
                <strong>
                  {selectedMap
                    ? `${selectedMap.mean_temp_c.toFixed(2)}°C`
                    : "—"}
                </strong>
              </div>

              <div className="stat-row">
                <span>Minimum</span>
                <strong>
                  {selectedMap
                    ? `${selectedMap.min_temp_c.toFixed(2)}°C`
                    : "—"}
                </strong>
              </div>

              <div className="stat-row">
                <span>Maximum</span>
                <strong>
                  {selectedMap
                    ? `${selectedMap.max_temp_c.toFixed(2)}°C`
                    : "—"}
                </strong>
              </div>
            </div>

            {selectedDepthData && (
              <div className="stats-card">
                <span className="eyebrow">RECONSTRUCTION SUMMARY</span>

                <div className="summary-depth">
                  {selectedDepthData.depth_m} m
                </div>

                <p>
                  CNN reconstruction generated from three surface-derived
                  predictors across the Bay of Bengal grid.
                </p>
              </div>
            )}
          </aside>
        </section>
                <section className="validation-section">
          <div className="validation-header">
            <div>
              <span className="eyebrow">MODEL VALIDATION</span>
              <h3>How well does OceanEmbed reconstruct?</h3>
            </div>

            <span className="validation-note">
              V1 trained and evaluated on Jan–Jun 2020
            </span>
          </div>

          <div className="validation-grid">
            <div className="validation-card">
              <span className="validation-label">GLORYS HOLDOUT</span>

              <div className="validation-metric">
                <strong>1.42°C</strong>
                <span>RMSE</span>
              </div>

              <div className="validation-submetric">
                Baseline: <strong>1.62°C</strong>
              </div>

              <div className="validation-improvement">
                12.3% RMSE reduction vs climatology
              </div>
            </div>

            <div className="validation-card">
              <span className="validation-label">CORA • MAY</span>

              <div className="validation-metric">
                <strong>0.975°C</strong>
                <span>RMSE</span>
              </div>

              <div className="validation-submetric">
                MAE: <strong>0.708°C</strong>
              </div>

              <div className="validation-submetric">
                Pooled correlation: <strong>0.994</strong>
              </div>
            </div>

            <div className="validation-card">
              <span className="validation-label">CORA • JUNE</span>

              <div className="validation-metric">
                <strong>1.218°C</strong>
                <span>RMSE</span>
              </div>

              <div className="validation-submetric">
                MAE: <strong>0.845°C</strong>
              </div>

              <div className="validation-submetric">
                Pooled correlation: <strong>0.988</strong>
              </div>
            </div>
          </div>

          <p className="validation-footnote">
            CORA is an independent observationally derived objective-analysis
            product used for external validation. These results represent the
            current V1 evaluation period and should not be interpreted as
            real-time performance.
          </p>
        </section>
      </main>
    </div>
  );
}
function TemperatureMap({ map, oceanMask }) {
  const rows = map.length;
  const cols = map[0]?.length || 0;

  if (!rows || !cols) {
    return (
      <div className="map-empty">
        <p>No map data available.</p>
      </div>
    );
  }

  // Find temperature range using ocean cells only.
  let min = Infinity;
  let max = -Infinity;

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      if (oceanMask?.[r]?.[c] && Number.isFinite(map[r][c])) {
        const value = map[r][c];

        if (value < min) min = value;
        if (value > max) max = value;
      }
    }
  }

  const legendValues = Array.from({ length: 5 }, (_, index) => {
    if (max === min) return min;
    return min + ((max - min) * index) / 4;
  });

  return (
    <>
      {/* Temperature field */}
      <div
        className="temperature-grid"
        style={{
          gridTemplateColumns: `repeat(${cols}, 1fr)`,
          gridTemplateRows: `repeat(${rows}, 1fr)`,
        }}
      >
        {map.flatMap((row, rowIndex) =>
          row.map((value, colIndex) => {
            const isOcean = oceanMask?.[rowIndex]?.[colIndex];

            // Land / masked cells.
            if (!isOcean) {
              return (
                <div
                  key={`${rowIndex}-${colIndex}`}
                  className="temperature-cell"
                  style={{
                    backgroundColor: "rgba(3, 15, 24, 0.95)",
                  }}
                />
              );
            }

            const normalized =
              max === min ? 0.5 : (value - min) / (max - min);

            const hue = 220 - normalized * 220;

            return (
              <div
                key={`${rowIndex}-${colIndex}`}
                className="temperature-cell"
                title={`${value.toFixed(2)}°C`}
                style={{
                  backgroundColor: `hsl(${hue}, 78%, 52%)`,
                }}
              />
            );
          })
        )}
      </div>

      {/* Temperature legend */}
      <div
        style={{
          marginTop: "18px",
          padding: "0 4px 4px",
        }}
      >
        <div
          style={{
            fontSize: "11px",
            letterSpacing: "0.08em",
            textTransform: "uppercase",
            opacity: 0.7,
            marginBottom: "8px",
          }}
        >
          Temperature (°C)
        </div>

        <div
          style={{
            height: "10px",
            borderRadius: "999px",
            background:
              "linear-gradient(90deg, hsl(220,78%,52%), hsl(165,78%,52%), hsl(110,78%,52%), hsl(55,78%,52%), hsl(0,78%,52%))",
          }}
        />

        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            marginTop: "6px",
            fontSize: "11px",
            opacity: 0.75,
          }}
        >
          {legendValues.map((value, index) => (
            <span key={index}>{value.toFixed(1)}°</span>
          ))}
        </div>
      </div>
    </>
  );
}

export default App;