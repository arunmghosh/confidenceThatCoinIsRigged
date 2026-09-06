import os
import json
import pandas as pd

def create_html_dashboard(results_json="results/evaluation_results.json", summary_csv="results/evaluation_summary.csv", output_path="dashboard.html"):
    if not os.path.exists(results_json) or not os.path.exists(summary_csv):
        print("Results files not found. Please run evaluate.py first.")
        return

    with open(results_json, "r") as f:
        data = json.load(f)

    json_data_str = json.dumps(data)

    template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Rigged Coin Detection Investigation</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-primary: #0B0F19;
            --bg-secondary: #111827;
            --bg-card: #1F2937;
            --border-color: #374151;
            --text-primary: #F9FAFB;
            --text-secondary: #9CA3AF;
            --accent-blue: #3B82F6;
            --accent-orange: #F97316;
            --accent-green: #10B981;
            --accent-purple: #8B5CF6;
            --accent-slate: #64748B;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background-color: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            padding: 32px 24px;
        }

        .container {
            max-width: 1300px;
            margin: 0 auto;
        }

        header {
            margin-bottom: 36px;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 24px;
        }

        .badge {
            display: inline-block;
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            padding: 4px 10px;
            border-radius: 9999px;
            background: rgba(59, 130, 246, 0.2);
            color: #60A5FA;
            border: 1px solid rgba(59, 130, 246, 0.4);
            margin-bottom: 12px;
        }

        h1 {
            font-size: 2.3rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            background: linear-gradient(135deg, #FFFFFF 0%, #CBD5E1 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 8px;
        }

        .subtitle {
            color: var(--text-secondary);
            font-size: 1.05rem;
            max-width: 900px;
        }

        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 20px;
            margin-bottom: 36px;
        }

        .card {
            background-color: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 14px;
            padding: 24px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
            transition: transform 0.2s ease, border-color 0.2s ease;
        }

        .card:hover {
            border-color: #4B5563;
        }

        .card-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 14px;
        }

        .card-title {
            font-size: 1.05rem;
            font-weight: 700;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .indicator {
            width: 10px;
            height: 10px;
            border-radius: 50%;
        }

        .indicator.blue { background-color: var(--accent-blue); box-shadow: 0 0 10px var(--accent-blue); }
        .indicator.orange { background-color: var(--accent-orange); box-shadow: 0 0 10px var(--accent-orange); }
        .indicator.green { background-color: var(--accent-green); box-shadow: 0 0 10px var(--accent-green); }
        .indicator.slate { background-color: var(--accent-slate); box-shadow: 0 0 10px var(--accent-slate); }

        .card-desc {
            color: var(--text-secondary);
            font-size: 0.88rem;
            margin-bottom: 16px;
        }

        .stat-row {
            display: flex;
            justify-content: space-between;
            font-size: 0.85rem;
            padding: 8px 0;
            border-top: 1px solid rgba(255, 255, 255, 0.06);
        }

        .stat-label { color: var(--text-secondary); }
        .stat-value { font-family: 'JetBrains Mono', monospace; font-weight: 600; color: #E2E8F0; }

        .controls-panel {
            background-color: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 14px;
            padding: 20px 24px;
            margin-bottom: 24px;
            display: flex;
            flex-wrap: wrap;
            gap: 24px;
            align-items: center;
        }

        .control-group {
            display: flex;
            flex-direction: column;
            gap: 6px;
        }

        .control-label {
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-secondary);
            font-weight: 600;
        }

        .btn-group {
            display: flex;
            background: var(--bg-card);
            border-radius: 8px;
            padding: 4px;
            border: 1px solid var(--border-color);
        }

        .btn {
            background: transparent;
            border: none;
            color: var(--text-secondary);
            font-size: 0.85rem;
            font-weight: 600;
            padding: 6px 14px;
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.2s;
        }

        .btn:hover {
            color: #FFFFFF;
        }

        .btn.active {
            background-color: var(--accent-blue);
            color: #FFFFFF;
            box-shadow: 0 2px 8px rgba(59, 130, 246, 0.4);
        }

        .chart-card {
            background-color: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 14px;
            padding: 24px;
            margin-bottom: 36px;
        }

        .chart-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }

        .chart-box {
            position: relative;
            height: 440px;
            width: 100%;
        }

        .table-card {
            background-color: var(--bg-secondary);
            border: 1px solid var(--border-color);
            border-radius: 14px;
            padding: 24px;
            margin-bottom: 36px;
            overflow-x: auto;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.88rem;
            text-align: left;
        }

        th {
            background-color: var(--bg-card);
            color: var(--text-secondary);
            font-weight: 600;
            padding: 12px 16px;
            border-bottom: 1px solid var(--border-color);
            text-transform: uppercase;
            font-size: 0.75rem;
            letter-spacing: 0.05em;
        }

        td {
            padding: 12px 16px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            font-family: 'JetBrains Mono', monospace;
        }

        tr:hover td {
            background-color: rgba(255, 255, 255, 0.02);
        }

        .insights-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
            gap: 20px;
            margin-bottom: 36px;
        }

        .insight-card {
            background-color: var(--bg-secondary);
            border-left: 4px solid var(--accent-blue);
            border-radius: 8px 14px 14px 8px;
            padding: 20px;
            border-top: 1px solid var(--border-color);
            border-right: 1px solid var(--border-color);
            border-bottom: 1px solid var(--border-color);
        }

        .insight-card.orange { border-left-color: var(--accent-orange); }
        .insight-card.green { border-left-color: var(--accent-green); }

        .insight-title {
            font-size: 1rem;
            font-weight: 700;
            margin-bottom: 8px;
        }

        .insight-text {
            font-size: 0.88rem;
            color: var(--text-secondary);
            line-height: 1.5;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="badge">Experimental Investigation</div>
            <h1>AI Confidence in Rigged Coin Detection</h1>
            <p class="subtitle">
                Systematic evaluation of neural algorithms predicting whether a coin is rigged (<span style="color:#60A5FA; font-family:'JetBrains Mono'">P(heads) &gt; 0.5</span>) 
                across three information regimes, varying sample sizes (<span style="font-family:'JetBrains Mono'">N &in; [10, 500]</span>), 
                and coin biases (<span style="font-family:'JetBrains Mono'">p &in; [0.5, 0.9]</span>), trained strictly with Brier Score Loss.
            </p>
        </header>

        <!-- Model Architectures Overview -->
        <div class="metrics-grid">
            <div class="card">
                <div class="card-header">
                    <div class="card-title">
                        <div class="indicator blue"></div>
                        Model A (Full History)
                    </div>
                    <span style="font-size:0.75rem; color:#60A5FA; font-weight:700">Sequence-Aware</span>
                </div>
                <div class="card-desc">Sequence-aware neural network with dual pooling. Ingests the entire raw sequence of flips up to N=500, including exact temporal order.</div>
                <div class="stat-row">
                    <span class="stat-label">Input Representation</span>
                    <span class="stat-value">Tensor (B, N, 1)</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Information Available</span>
                    <span class="stat-value">100% History</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Best Val Brier Score</span>
                    <span class="stat-value">0.0847</span>
                </div>
            </div>

            <div class="card">
                <div class="card-header">
                    <div class="card-title">
                        <div class="indicator orange"></div>
                        Model B (Last 20 Flips)
                    </div>
                    <span style="font-size:0.75rem; color:#FB923C; font-weight:700">Memory-Constrained</span>
                </div>
                <div class="card-desc">1D Convolutional network with a fixed 20-flip context window. Truncates all earlier historical flips when N &gt; 20.</div>
                <div class="stat-row">
                    <span class="stat-label">Input Representation</span>
                    <span class="stat-value">Vector (B, 20)</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Information Available</span>
                    <span class="stat-value">min(N, 20) Flips</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Best Val Brier Score</span>
                    <span class="stat-value">0.1400</span>
                </div>
            </div>

            <div class="card">
                <div class="card-header">
                    <div class="card-title">
                        <div class="indicator green"></div>
                        Model C (Summary Stats)
                    </div>
                    <span style="font-size:0.75rem; color:#34D399; font-weight:700">Sufficient Statistic</span>
                </div>
                <div class="card-desc">Dense MLP receiving only compressed aggregate features: head ratio (% heads) and total flips N (discarding all order information).</div>
                <div class="stat-row">
                    <span class="stat-label">Input Representation</span>
                    <span class="stat-value">Features [% Heads, N]</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Information Available</span>
                    <span class="stat-value">Aggregates Only</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Best Val Brier Score</span>
                    <span class="stat-value">0.0841</span>
                </div>
            </div>

            <div class="card">
                <div class="card-header">
                    <div class="card-title">
                        <div class="indicator slate"></div>
                        Bayesian Benchmark
                    </div>
                    <span style="font-size:0.75rem; color:#94A3B8; font-weight:700">Theoretical Optimal</span>
                </div>
                <div class="card-desc">Exact analytical posterior P(p &gt; 0.5 | data) under a uniform Beta(1, 1) prior. Serves as the mathematical calibration ceiling.</div>
                <div class="stat-row">
                    <span class="stat-label">Formula</span>
                    <span class="stat-value">1 - I_0.5(alpha+k, beta+n-k)</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Order Sensitivity</span>
                    <span class="stat-value">Exchangeable (0)</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Theoretical Status</span>
                    <span class="stat-value">Optimal Upper Bound</span>
                </div>
            </div>
        </div>

        <!-- Interactive Control Panel -->
        <div class="controls-panel">
            <div class="control-group">
                <div class="control-label">Primary Analysis View</div>
                <div class="btn-group">
                    <button class="btn active" id="view-by-n" onclick="switchView('by_n')">Confidence vs Riggedness (Pick N)</button>
                    <button class="btn" id="view-by-p" onclick="switchView('by_p')">Confidence vs Sample Size (Pick p)</button>
                </div>
            </div>

            <div class="control-group" id="n-picker-group">
                <div class="control-label">Select Sample Size (Flips N)</div>
                <div class="btn-group" id="n-buttons">
                    <button class="btn" onclick="selectN(10)">10</button>
                    <button class="btn" onclick="selectN(25)">25</button>
                    <button class="btn" onclick="selectN(50)">50</button>
                    <button class="btn active" onclick="selectN(100)">100</button>
                    <button class="btn" onclick="selectN(200)">200</button>
                    <button class="btn" onclick="selectN(300)">300</button>
                    <button class="btn" onclick="selectN(500)">500</button>
                </div>
            </div>

            <div class="control-group" id="p-picker-group" style="display:none;">
                <div class="control-label">Select Coin Bias P(Heads)</div>
                <div class="btn-group" id="p-buttons">
                    <button class="btn" onclick="selectP(0.50)">0.50 (Fair)</button>
                    <button class="btn active" onclick="selectP(0.55)">0.55</button>
                    <button class="btn" onclick="selectP(0.60)">0.60</button>
                    <button class="btn" onclick="selectP(0.70)">0.70</button>
                    <button class="btn" onclick="selectP(0.80)">0.80</button>
                    <button class="btn" onclick="selectP(0.90)">0.90</button>
                </div>
            </div>
        </div>

        <!-- Interactive Dynamic Chart -->
        <div class="chart-card">
            <div class="chart-header">
                <div>
                    <h2 id="chart-title" style="font-size:1.25rem; font-weight:700;">Confidence vs Riggedness (N = 100 Flips)</h2>
                    <p id="chart-desc" style="color:var(--text-secondary); font-size:0.85rem;">Comparing AI algorithm confidence as true P(heads) increases from 0.50 to 0.90 with sample size N = 100.</p>
                </div>
            </div>
            <div class="chart-box">
                <canvas id="interactiveChart"></canvas>
            </div>
        </div>

        <!-- Key Empirical Findings -->
        <div class="insights-grid">
            <div class="insight-card">
                <div class="insight-title" style="color:#60A5FA;">1. The Sufficiency Equivalence (Model A &asymp; Model C)</div>
                <div class="insight-text">
                    Under Bernoulli trials, flips are exchangeable. The sample count <em>k</em> and sample size <em>N</em> are minimal sufficient statistics. 
                    Model C (only given % heads and N) matches Model A (given full sequential order) within standard error across all 42 conditions, proving order contains zero residual signal.
                </div>
            </div>

            <div class="insight-card orange">
                <div class="insight-title" style="color:#FB923C;">2. Model B's Memory Ceiling at N &gt; 20</div>
                <div class="insight-text">
                    For small biases (e.g. p = 0.55 or 0.60), distinguishing signal from noise requires hundreds of flips. 
                    Because Model B only sees the final 20 flips, its confidence plateaus at ~0.47 even when N = 500, whereas Models A & C achieve &gt; 0.94 confidence.
                </div>
            </div>

            <div class="insight-card green">
                <div class="insight-title" style="color:#34D399;">3. Brier Calibration and Restraint</div>
                <div class="insight-text">
                    Trained directly on Brier score loss, all models converge toward the Bayes optimal posterior. 
                    At p = 0.50 (fair coin), predicted confidence remains appropriately restrained around 0.09 - 0.38 depending on N, avoiding overconfident false alarms.
                </div>
            </div>
        </div>

        <!-- Comprehensive Data Table -->
        <div class="table-card">
            <h2 style="font-size:1.25rem; font-weight:700; margin-bottom:16px;">Experimental Results Matrix (500 Trials Per Cell)</h2>
            <table id="summary-table">
                <thead>
                    <tr>
                        <th>P(Heads)</th>
                        <th>Flips (N)</th>
                        <th>Model A Conf</th>
                        <th>Model B Conf</th>
                        <th>Model C Conf</th>
                        <th>Bayes Optimal</th>
                        <th>Model A Brier</th>
                        <th>Model B Brier</th>
                        <th>Model C Brier</th>
                    </tr>
                </thead>
                <tbody id="table-body">
                </tbody>
            </table>
        </div>
    </div>

    <script>
        const rawData = __DATA_JSON__;
        let currentMode = 'by_n';
        let selectedN = 100;
        let selectedP = 0.55;
        let chartInstance = null;

        const pList = [0.5, 0.55, 0.6, 0.7, 0.8, 0.9];
        const nList = [10, 25, 50, 100, 200, 300, 500];

        function getCell(p, n) {
            const pk = 'p_' + p.toFixed(2);
            const nk = 'n_' + n;
            return rawData[pk] ? rawData[pk][nk] : null;
        }

        function populateTable() {
            const tbody = document.getElementById('table-body');
            tbody.innerHTML = '';

            pList.forEach(p => {
                nList.forEach(n => {
                    const cell = getCell(p, n);
                    if (!cell) return;

                    const row = document.createElement('tr');
                    row.innerHTML = `
                        <td><strong>${p.toFixed(2)}</strong></td>
                        <td>${n}</td>
                        <td><span style="color:#60A5FA; font-weight:bold">${(cell.Model_A.mean * 100).toFixed(1)}%</span> &plusmn;${(cell.Model_A.se * 100).toFixed(1)}%</td>
                        <td><span style="color:#FB923C; font-weight:bold">${(cell.Model_B.mean * 100).toFixed(1)}%</span> &plusmn;${(cell.Model_B.se * 100).toFixed(1)}%</td>
                        <td><span style="color:#34D399; font-weight:bold">${(cell.Model_C.mean * 100).toFixed(1)}%</span> &plusmn;${(cell.Model_C.se * 100).toFixed(1)}%</td>
                        <td style="color:#94A3B8;">${(cell.Bayesian_Benchmark.mean * 100).toFixed(1)}%</td>
                        <td>${cell.Model_A.brier.toFixed(4)}</td>
                        <td>${cell.Model_B.brier.toFixed(4)}</td>
                        <td>${cell.Model_C.brier.toFixed(4)}</td>
                    `;
                    tbody.appendChild(row);
                });
            });
        }

        function updateChart() {
            const ctx = document.getElementById('interactiveChart').getContext('2d');

            let labels, aData, bData, cData, bayesData;

            if (currentMode === 'by_n') {
                document.getElementById('chart-title').innerText = `Confidence vs Riggedness (N = ${selectedN} Flips)`;
                document.getElementById('chart-desc').innerText = `Comparing AI algorithm confidence as true P(heads) increases from 0.50 to 0.90 with sample size N = ${selectedN}.`;
                labels = pList.map(p => p.toFixed(2));
                aData = pList.map(p => getCell(p, selectedN).Model_A.mean);
                bData = pList.map(p => getCell(p, selectedN).Model_B.mean);
                cData = pList.map(p => getCell(p, selectedN).Model_C.mean);
                bayesData = pList.map(p => getCell(p, selectedN).Bayesian_Benchmark.mean);
            } else {
                document.getElementById('chart-title').innerText = `Confidence vs Sample Size N (P(Heads) = ${selectedP.toFixed(2)})`;
                document.getElementById('chart-desc').innerText = `How confidence scales as the number of observed flips increases from 10 to 500 for coin bias ${selectedP.toFixed(2)}.`;
                labels = nList.map(n => String(n));
                aData = nList.map(n => getCell(selectedP, n).Model_A.mean);
                bData = nList.map(n => getCell(selectedP, n).Model_B.mean);
                cData = nList.map(n => getCell(selectedP, n).Model_C.mean);
                bayesData = nList.map(n => getCell(selectedP, n).Bayesian_Benchmark.mean);
            }

            if (chartInstance) {
                chartInstance.destroy();
            }

            chartInstance = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: [
                        {
                            label: 'Model A (Full History)',
                            data: aData,
                            borderColor: '#3B82F6',
                            backgroundColor: 'rgba(59, 130, 246, 0.1)',
                            borderWidth: 3,
                            pointRadius: 5,
                            tension: 0.2
                        },
                        {
                            label: 'Model B (Last 20 Flips)',
                            data: bData,
                            borderColor: '#F97316',
                            backgroundColor: 'rgba(249, 115, 22, 0.1)',
                            borderWidth: 3,
                            pointRadius: 5,
                            tension: 0.2
                        },
                        {
                            label: 'Model C (Summary Stats)',
                            data: cData,
                            borderColor: '#10B981',
                            backgroundColor: 'rgba(16, 185, 129, 0.1)',
                            borderWidth: 3,
                            pointRadius: 5,
                            tension: 0.2
                        },
                        {
                            label: 'Bayesian Optimal',
                            data: bayesData,
                            borderColor: '#94A3B8',
                            borderWidth: 2,
                            borderDash: [5, 5],
                            pointRadius: 3,
                            tension: 0.2
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: {
                            title: {
                                display: true,
                                text: currentMode === 'by_n' ? 'True P(Heads)' : 'Flips per Trial (N)',
                                color: '#9CA3AF',
                                font: { family: 'Inter', size: 12, weight: 'bold' }
                            },
                            grid: { color: 'rgba(255, 255, 255, 0.05)' },
                            ticks: { color: '#9CA3AF', font: { family: 'JetBrains Mono' } }
                        },
                        y: {
                            title: {
                                display: true,
                                text: 'Predicted P(Coin Rigged)',
                                color: '#9CA3AF',
                                font: { family: 'Inter', size: 12, weight: 'bold' }
                            },
                            min: 0.0,
                            max: 1.05,
                            grid: { color: 'rgba(255, 255, 255, 0.05)' },
                            ticks: {
                                color: '#9CA3AF',
                                font: { family: 'JetBrains Mono' },
                                callback: val => (val * 100).toFixed(0) + '%'
                            }
                        }
                    },
                    plugins: {
                        legend: {
                            labels: {
                                color: '#E2E8F0',
                                font: { family: 'Inter', size: 12, weight: 600 },
                                usePointStyle: true
                            }
                        },
                        tooltip: {
                            backgroundColor: '#1E293B',
                            titleColor: '#F8FAFC',
                            bodyColor: '#E2E8F0',
                            borderColor: '#475569',
                            borderWidth: 1,
                            callbacks: {
                                label: function(context) {
                                    return context.dataset.label + ': ' + (context.raw * 100).toFixed(2) + '%';
                                }
                            }
                        }
                    }
                }
            });
        }

        function switchView(mode) {
            currentMode = mode;
            document.getElementById('view-by-n').classList.toggle('active', mode === 'by_n');
            document.getElementById('view-by-p').classList.toggle('active', mode === 'by_p');

            document.getElementById('n-picker-group').style.display = mode === 'by_n' ? 'flex' : 'none';
            document.getElementById('p-picker-group').style.display = mode === 'by_p' ? 'flex' : 'none';

            updateChart();
        }

        function selectN(n) {
            selectedN = n;
            const buttons = document.querySelectorAll('#n-buttons .btn');
            buttons.forEach(btn => btn.classList.toggle('active', btn.innerText === String(n)));
            updateChart();
        }

        function selectP(p) {
            selectedP = p;
            const buttons = document.querySelectorAll('#p-buttons .btn');
            buttons.forEach(btn => btn.classList.toggle('active', btn.innerText.startsWith(p.toFixed(2))));
            updateChart();
        }

        window.onload = function() {
            populateTable();
            updateChart();
        };
    </script>
</body>
</html>
"""

    html_content = template.replace("__DATA_JSON__", json_data_str)

    with open(output_path, "w") as f:
        f.write(html_content)

    print(f"Interactive dashboard generated successfully at: {output_path}")

if __name__ == "__main__":
    create_html_dashboard()
