// ── Chart Instances ──
let tokensChartInstance = null;
let latencyChartInstance = null;
let rewardChartInstance = null;
let callsChartInstance = null;
let categoryChartInstance = null;

// ── Cumulative Live Stats ──
let cumQueries = 0;
let cumLatency = 0;
let cumTokens = 0;

const ALL_ACTIONS = ["3b_direct", "3b_cot", "7b_direct", "7b_cot", "7b_got", "7b_tot"];
const ALL_AGENTS = ["3b_direct", "3b_cot", "7b_direct", "7b_cot", "7b_got", "7b_tot", "cascaderoute"];

let strategySelectionCounts = {};
let strategyEvaluatedCorrect = {};
let strategyEvaluatedTotal = {};
let strategySumTokens = {};
let strategySumLatency = {};

ALL_ACTIONS.forEach(a => {
    strategySelectionCounts[a] = 0;
    strategyEvaluatedCorrect[a] = 0;
    strategyEvaluatedTotal[a] = 0;
    strategySumTokens[a] = 0;
    strategySumLatency[a] = 0;
});

// Live Session Stats Map for Agent Performance Summary
let liveAgentStatsMap = {};
ALL_AGENTS.forEach(a => {
    liveAgentStatsMap[a] = {
        agent: a,
        queries: 0,
        correct: 0,
        tokens: 0,
        latency: 0,
        calls: 0,
        reward: 0
    };
});

// ── Color Palette ──
const STRATEGY_COLORS = {
    '3b_direct': '#0984e3',
    '3b_cot': '#74b9ff',
    '7b_direct': '#6c5ce7',
    '7b_cot': '#a29bfe',
    '7b_got': '#e84393',
    '7b_tot': '#fdcb6e',
    'cascaderoute': '#00b894',
    'direct': '#0984e3',
    'cot': '#6c5ce7',
    'got': '#e84393',
    'tot': '#fdcb6e',
};

const AGENT_COLORS = {
    '3b_direct': '#0984e3', '3b_cot': '#74b9ff',
    '7b_direct': '#6c5ce7', '7b_cot': '#a29bfe',
    '7b_got': '#e84393', '7b_tot': '#fdcb6e',
    'cascaderoute': '#00b894', 'always_direct': '#95a5a6',
    'always_cot': '#bdc3c7', 'always_got': '#7f8c8d', 'always_tot': '#636e72',
    'random': '#fdcb6e', 'rule_based': '#e17055',
    'epsilon_greedy': '#74b9ff', 'ucb1': '#0984e3',
    'linucb': '#e84393', 'thompson': '#6c5ce7', 'dqn': '#00b894',
};

// ── Chart.js Defaults ──
Chart.defaults.color = '#8888aa';
Chart.defaults.borderColor = 'rgba(255,255,255,0.06)';
Chart.defaults.font.family = "'Inter', sans-serif";

// ── Global History Data Store ──
let globalHistoryData = null;

// ── Load History on Page Load ──
document.addEventListener('DOMContentLoaded', loadHistory);

async function loadHistory() {
    try {
        const resp = await fetch('/api/history');
        if (!resp.ok) return;
        const data = await resp.json();
        globalHistoryData = data;

        // Update header stats
        document.getElementById('statRows').textContent = data.total_rows ? data.total_rows.toLocaleString() : '3,712';
        if (data.agent_stats && data.agent_stats.length > 0) {
            const best = data.agent_stats.reduce((a, b) =>
                (a.accuracy || 0) > (b.accuracy || 0) ? a : b
            );
            document.getElementById('statBest').textContent = best.agent.toUpperCase();
        }
        if (data.oracle_accuracy) {
            document.getElementById('statOracle').textContent = data.oracle_accuracy + '%';
        }

        // Populate agent table initially with baseline history
        populateAgentTable(getMergedAgentStats());
    } catch (e) {
        console.log('No history data available:', e);
    }
}

function getMergedAgentStats() {
    if (!globalHistoryData || !globalHistoryData.agent_stats) {
        return ALL_AGENTS.map(a => ({
            agent: a,
            accuracy: liveAgentStatsMap[a].queries > 0 ? (liveAgentStatsMap[a].correct / liveAgentStatsMap[a].queries * 100).toFixed(1) : 0,
            avg_tokens: liveAgentStatsMap[a].queries > 0 ? Math.round(liveAgentStatsMap[a].tokens / liveAgentStatsMap[a].queries) : 0,
            median_latency: liveAgentStatsMap[a].queries > 0 ? (liveAgentStatsMap[a].latency / liveAgentStatsMap[a].queries).toFixed(2) : '0.00',
            avg_reward: liveAgentStatsMap[a].queries > 0 ? (liveAgentStatsMap[a].reward / liveAgentStatsMap[a].queries).toFixed(4) : 0,
            avg_llm_calls: liveAgentStatsMap[a].queries > 0 ? (liveAgentStatsMap[a].calls / liveAgentStatsMap[a].queries).toFixed(1) : 1
        }));
    }

    const baseStatsMap = {};
    globalHistoryData.agent_stats.forEach(s => {
        baseStatsMap[s.agent] = s;
    });

    return ALL_AGENTS.map(a => {
        const live = liveAgentStatsMap[a];
        const base = baseStatsMap[a] || {};

        if (live.queries > 0) {
            return {
                agent: a,
                accuracy: (live.correct / live.queries * 100).toFixed(1),
                avg_tokens: Math.round(live.tokens / live.queries),
                median_latency: (live.latency / live.queries).toFixed(2),
                avg_reward: (live.reward / live.queries).toFixed(4),
                avg_llm_calls: (live.calls / live.queries).toFixed(1)
            };
        } else {
            return {
                agent: a,
                accuracy: base.accuracy !== undefined ? base.accuracy : 0,
                avg_tokens: base.avg_tokens !== undefined ? base.avg_tokens : 0,
                median_latency: base.median_latency !== undefined ? base.median_latency : '0.00',
                avg_reward: base.avg_reward !== undefined ? base.avg_reward : '0.0000',
                avg_llm_calls: base.avg_llm_calls !== undefined ? base.avg_llm_calls : 1
            };
        }
    });
}

function populateAgentTable(stats) {
    const tbody = document.getElementById('agentTableBody');
    if (!tbody || !stats) return;
    tbody.innerHTML = '';

    function getAgentEfficiencyScore(s) {
        const acc = parseFloat(s.accuracy) || 0;
        const tok = parseFloat(s.avg_tokens) || 9999;
        const lat = parseFloat(s.median_latency) || 999;
        return acc * 10000 - tok - lat * 10;
    }

    const best = stats.reduce((a, b) => getAgentEfficiencyScore(a) > getAgentEfficiencyScore(b) ? a : b, stats[0]);

    stats.forEach(s => {
        const row = document.createElement('tr');
        const isBest = s.agent === best.agent;
        if (isBest) row.className = 'best-row';
        if (s.agent === 'cascaderoute' || s.agent === 'linucb') row.className += ' agent-row-linucb';

        row.innerHTML = `
            <td>${isBest ? '⭐ ' : ''}<strong>${s.agent.toUpperCase()}</strong></td>
            <td>${s.accuracy}%</td>
            <td>${s.avg_tokens}</td>
            <td>${s.median_latency}s</td>
            <td>${s.avg_reward !== null ? s.avg_reward : '---'}</td>
            <td>${s.avg_llm_calls}</td>
        `;
        tbody.appendChild(row);
    });
}

// ── Run Analysis ──
async function runAnalysis() {
    const question = document.getElementById('questionInput').value.trim();
    if (!question) {
        alert('Please enter a question!');
        return;
    }

    const category = document.getElementById('categorySelect').value;
    const expectedAnswer = document.getElementById('expectedAnswer').value.trim();
    const btn = document.getElementById('analyzeBtn');
    const spinner = document.getElementById('analyzeSpinner');

    // Disable button
    btn.disabled = true;
    spinner.classList.remove('hidden');
    document.querySelector('.btn-text').textContent = 'Analyzing...';

    try {
        // Run both requests in parallel
        const [analyzeResp, recResp] = await Promise.all([
            fetch('/api/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question, category, expected_answer: expectedAnswer }),
            }),
            fetch('/api/recommend', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question, category }),
            }),
        ]);

        const analyzeData = await analyzeResp.json();
        const recData = await recResp.json();

        if (analyzeData.error) throw new Error(analyzeData.error);
        if (recData.error) throw new Error(recData.error);

        // Show results
        renderResults(analyzeData.results);
        renderRecommendation(recData);
        renderEfficiency(analyzeData.results, recData.recommended_strategy);
        updateCumulativeStats(analyzeData.results, recData.recommended_strategy);
        renderLiveCharts(analyzeData.results);
        renderTraces(analyzeData.results);

    } catch (e) {
        alert('Error: ' + e.message + '\n\nMake sure local Ollama is running!');
    } finally {
        btn.disabled = false;
        spinner.classList.add('hidden');
        document.querySelector('.btn-text').textContent = 'Analyze All Strategies';
    }
}

function renderResults(results) {
    const panel = document.getElementById('resultsPanel');
    panel.classList.remove('hidden');
    panel.classList.add('show');

    const tbody = document.getElementById('resultsBody');
    tbody.innerHTML = '';

    const bestReward = Math.max(...results.map(r => r.reward));

    results.forEach(r => {
        const row = document.createElement('tr');
        if (r.reward === bestReward) row.className = 'best-row';

        let correctHtml;
        if (r.correct === true) correctHtml = '<span class="correct-badge">CORRECT</span>';
        else if (r.correct === false) correctHtml = '<span class="incorrect-badge">WRONG</span>';
        else correctHtml = '<span class="na-badge">N/A</span>';

        row.innerHTML = `
            <td style="font-weight:600; color:${STRATEGY_COLORS[r.strategy]}">${r.strategy.toUpperCase()}</td>
            <td>${r.answer || '(no answer)'}</td>
            <td>${correctHtml}</td>
            <td>${r.total_tokens.toLocaleString()}</td>
            <td>${r.latency}s</td>
            <td>${r.n_llm_calls}</td>
            <td style="font-weight:600">${r.reward}</td>
        `;
        tbody.appendChild(row);
    });
}

function updateCumulativeStats(results, recommendedStrategy) {
    cumQueries++;
    const pickedResult = results.find(r => r.strategy === recommendedStrategy);
    if (pickedResult) {
        cumLatency += pickedResult.latency;
        cumTokens += pickedResult.total_tokens;
        strategySelectionCounts[recommendedStrategy] = (strategySelectionCounts[recommendedStrategy] || 0) + 1;
        
        // Track live stats for cascaderoute
        liveAgentStatsMap['cascaderoute'].queries += 1;
        if (pickedResult.correct === true) liveAgentStatsMap['cascaderoute'].correct += 1;
        liveAgentStatsMap['cascaderoute'].tokens += pickedResult.total_tokens;
        liveAgentStatsMap['cascaderoute'].latency += pickedResult.latency;
        liveAgentStatsMap['cascaderoute'].calls += (pickedResult.n_llm_calls || 1);
        liveAgentStatsMap['cascaderoute'].reward += (pickedResult.reward || 0);
    }

    results.forEach(r => {
        const s = r.strategy;
        strategyEvaluatedTotal[s] = (strategyEvaluatedTotal[s] || 0) + 1;
        if (r.correct === true) {
            strategyEvaluatedCorrect[s] = (strategyEvaluatedCorrect[s] || 0) + 1;
        }
        strategySumTokens[s] = (strategySumTokens[s] || 0) + r.total_tokens;
        strategySumLatency[s] = (strategySumLatency[s] || 0) + r.latency;

        // Update liveAgentStatsMap for individual strategy
        if (liveAgentStatsMap[s]) {
            liveAgentStatsMap[s].queries += 1;
            if (r.correct === true) liveAgentStatsMap[s].correct += 1;
            liveAgentStatsMap[s].tokens += r.total_tokens;
            liveAgentStatsMap[s].latency += r.latency;
            liveAgentStatsMap[s].calls += (r.n_llm_calls || 1);
            liveAgentStatsMap[s].reward += (r.reward || 0);
        }
    });
    
    document.getElementById('cumQueries').textContent = cumQueries;
    document.getElementById('cumLatency').textContent = (cumLatency / cumQueries).toFixed(1) + 's';
    document.getElementById('cumTokens').textContent = Math.round(cumTokens / cumQueries).toLocaleString();
    
    const tbody = document.getElementById('cumTableBody');
    tbody.innerHTML = '';

    ALL_ACTIONS.forEach(strat => {
        const row = document.createElement('tr');
        const timesPicked = strategySelectionCounts[strat] || 0;
        const totalEval = strategyEvaluatedTotal[strat] || 0;
        const corrCount = strategyEvaluatedCorrect[strat] || 0;
        const pct = totalEval > 0 ? Math.round((corrCount / totalEval) * 100) : 0;

        row.innerHTML = `
            <td><strong style="color:${STRATEGY_COLORS[strat]||'#fff'}">${strat.toUpperCase()}</strong></td>
            <td>${timesPicked}</td>
            <td><span class="${pct > 50 ? 'correct-badge' : 'incorrect-badge'}">${pct}%</span> (${corrCount}/${totalEval})</td>
        `;
        tbody.appendChild(row);
    });

    // Dynamically update the Agent Performance Summary table with live session metrics
    populateAgentTable(getMergedAgentStats());
}

function renderEfficiency(results, recommendedStrategy) {
    const panel = document.getElementById('efficiencyPanel');
    panel.classList.remove('hidden');
    
    const simple = results.find(r => r.strategy === '3b_direct') || {};
    const complex = results.find(r => r.strategy === '7b_tot') || {};
    const picked = results.find(r => r.strategy === recommendedStrategy) || {};
    
    document.getElementById('effSimpleStatus').textContent = (simple.correct === true) ? 'CORRECT' : (simple.correct === false ? 'WRONG' : 'N/A');
    document.getElementById('effSimpleTokens').textContent = simple.total_tokens || 0;
    
    document.getElementById('effComplexStatus').textContent = (complex.correct === true) ? 'CORRECT' : (complex.correct === false ? 'WRONG' : 'N/A');
    document.getElementById('effComplexTokens').textContent = complex.total_tokens || 0;
    
    document.getElementById('effRLPick').textContent = (recommendedStrategy || '').toUpperCase();
    document.getElementById('effRLStatus').textContent = (picked.correct === true) ? 'CORRECT' : (picked.correct === false ? 'WRONG' : 'N/A');
    
    const savings = (complex.total_tokens || 0) - (picked.total_tokens || 0);
    document.getElementById('effSavings').textContent = savings > 0 ? savings.toLocaleString() : 0;
}

function renderRecommendation(data) {
    const panel = document.getElementById('recommendPanel');
    panel.classList.remove('hidden');
    panel.classList.add('show');

    document.getElementById('recStrategy').textContent = data.recommended_strategy.toUpperCase();

    const barsContainer = document.getElementById('confidenceBars');
    barsContainer.innerHTML = '';

    ALL_ACTIONS.forEach(s => {
        const conf = data.confidence[s] !== undefined ? data.confidence[s] : 0;
        const wrap = document.createElement('div');
        wrap.className = 'conf-bar-wrap';
        wrap.innerHTML = `
            <span class="conf-label">${s.toUpperCase()}</span>
            <div class="conf-bar-bg">
                <div class="conf-bar-fill" style="width: ${conf}%; background:${STRATEGY_COLORS[s] || '#6c5ce7'};"></div>
            </div>
            <span class="conf-value">${conf}%</span>
        `;
        barsContainer.appendChild(wrap);
    });
}

function renderLiveCharts(results) {
    const panel = document.getElementById('chartsPanel');
    panel.classList.remove('hidden');
    panel.classList.add('show');

    const labels = results.map(r => r.strategy.toUpperCase());
    const colors = results.map(r => STRATEGY_COLORS[r.strategy] || '#6c5ce7');

    // Tokens Chart
    const tokensCtx = document.getElementById('tokensChart').getContext('2d');
    if (tokensChartInstance) tokensChartInstance.destroy();
    tokensChartInstance = new Chart(tokensCtx, {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                label: 'Total Tokens',
                data: results.map(r => r.total_tokens),
                backgroundColor: colors.map(c => c + '99'),
                borderColor: colors,
                borderWidth: 2,
                borderRadius: 6,
            }]
        },
        options: {
            responsive: true,
            plugins: { legend: { display: false } },
            scales: {
                y: { beginAtZero: true, grid: { color: 'rgba(255,255,255,0.04)' } },
                x: { grid: { display: false } },
            },
        },
    });

    // Latency Chart
    const latCtx = document.getElementById('latencyChart').getContext('2d');
    if (latencyChartInstance) latencyChartInstance.destroy();
    latencyChartInstance = new Chart(latCtx, {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                label: 'Latency (s)',
                data: results.map(r => r.latency),
                backgroundColor: colors.map(c => c + '99'),
                borderColor: colors,
                borderWidth: 2,
                borderRadius: 6,
            }]
        },
        options: {
            responsive: true,
            plugins: { legend: { display: false } },
            scales: {
                y: { beginAtZero: true, grid: { color: 'rgba(255,255,255,0.04)' } },
                x: { grid: { display: false } },
            },
        },
    });

    // Reward Chart
    const rewCtx = document.getElementById('rewardChart').getContext('2d');
    if (rewardChartInstance) rewardChartInstance.destroy();
    rewardChartInstance = new Chart(rewCtx, {
        type: 'doughnut',
        data: {
            labels,
            datasets: [{
                data: results.map(r => Math.max(r.reward, 0)),
                backgroundColor: colors,
                borderWidth: 0,
            }]
        },
        options: {
            responsive: true,
            plugins: { legend: { position: 'bottom' } },
        },
    });

    // Calls Chart
    const callsCtx = document.getElementById('callsChart').getContext('2d');
    if (callsChartInstance) callsChartInstance.destroy();
    callsChartInstance = new Chart(callsCtx, {
        type: 'bar',
        data: {
            labels,
            datasets: [{
                label: 'LLM Calls',
                data: results.map(r => r.n_llm_calls),
                backgroundColor: colors.map(c => c + '99'),
                borderColor: colors,
                borderWidth: 2,
                borderRadius: 6,
            }]
        },
        options: {
            responsive: true,
            plugins: { legend: { display: false } },
            scales: {
                y: { beginAtZero: true, grid: { color: 'rgba(255,255,255,0.04)' } },
                x: { grid: { display: false } },
            },
        },
    });
}

function renderTraces(results) {
    const panel = document.getElementById('tracesPanel');
    panel.classList.remove('hidden');
    panel.classList.add('show');

    const grid = document.getElementById('tracesGrid');
    grid.innerHTML = '';

    results.forEach(r => {
        const card = document.createElement('div');
        card.className = 'trace-card';

        const header = document.createElement('div');
        header.className = `trace-header ${r.strategy}`;
        header.style.borderLeft = `4px solid ${STRATEGY_COLORS[r.strategy] || '#6c5ce7'}`;
        header.innerHTML = `
            <span>${r.strategy.toUpperCase()} (${r.n_llm_calls} call${r.n_llm_calls > 1 ? 's' : ''})</span>
            <span class="trace-toggle">▼</span>
        `;

        const body = document.createElement('div');
        body.className = 'trace-body hidden';
        body.textContent = r.raw_text || r.error || '(empty response)';

        header.addEventListener('click', () => {
            body.classList.toggle('hidden');
            header.querySelector('.trace-toggle').classList.toggle('open');
        });

        card.appendChild(header);
        card.appendChild(body);
        grid.appendChild(card);
    });
}

// ── Allow Enter key to submit ──
document.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && e.ctrlKey) {
        runAnalysis();
    }
});
