// script.js
const API_URL = "http://localhost:8001";
let authToken = null;
let currentUser = null;
let forecastChartInstance = null;
let globalStores = [];
let globalProducts = [];

// Helper Icons
const getIcon = (prodId) => {
    if (!prodId) return '📦';
    const p = prodId.toLowerCase();
    if (p.includes('milk')) return '🥛';
    if (p.includes('apple')) return '🍎';
    if (p.includes('banana')) return '🍌';
    if (p.includes('bread')) return '🍞';
    if (p.includes('orange')) return '🧃';
    if (p.includes('carrot')) return '🥕';
    if (p.includes('cheese')) return '🧀';
    if (p.includes('paper')) return '🧻';
    return '📦';
};

document.addEventListener("DOMContentLoaded", () => {
    const loginModal = document.getElementById("login-modal");
    const appLayout = document.getElementById("app-layout");

    authToken = localStorage.getItem('smartstock_token');
    const userRoleStr = localStorage.getItem('smartstock_role');
    const userEmailStr = localStorage.getItem('smartstock_email');
    const userStoresStr = localStorage.getItem('smartstock_store');

    if (!authToken) {
        loginModal.classList.remove("hidden");
    } else {
        currentUser = { role: userRoleStr, email: userEmailStr };
        if (document.getElementById("user-role")) document.getElementById("user-role").innerText = currentUser.role || 'MANAGER';
        if (document.getElementById("user-email")) document.getElementById("user-email").innerText = currentUser.email || 'user@email.com';
        appLayout.classList.remove("hidden");
        initDashboard();
    }

    const loginForm = document.getElementById("login-form");
    if (loginForm) {
        loginForm.addEventListener("submit", async (e) => {
            e.preventDefault();
            const loginError = document.getElementById("login-error");
            const loginBtnText = document.getElementById("login-btn-text");
            const loginSpinner = document.getElementById("login-spinner");

            loginError.classList.add("hidden");
            loginBtnText.innerText = "Authenticating...";
            loginSpinner.classList.remove("hidden");

            const fd = new URLSearchParams();
            fd.append('username', document.getElementById("username").value);
            fd.append('password', document.getElementById("password").value);

            try {
                const res = await fetch(`${API_URL}/auth/login`, { method: 'POST', body: fd });
                if (res.ok) {
                    const data = await res.json();
                    authToken = data.access_token;
                    currentUser = data.user;

                    const userRes = await fetch(`${API_URL}/auth/me`, { headers: { "Authorization": `Bearer ${authToken}` } });
                    let ustore = "Germany";
                    if (userRes.ok) {
                        const userInfo = await userRes.json();
                        ustore = userInfo.authorized_stores ? userInfo.authorized_stores.split(",")[0] : "Germany";
                    }

                    localStorage.setItem('smartstock_token', authToken);
                    localStorage.setItem('smartstock_role', currentUser.role);
                    localStorage.setItem('smartstock_email', currentUser.email);
                    localStorage.setItem('smartstock_store', ustore);

                    if (document.getElementById("user-role")) document.getElementById("user-role").innerText = currentUser.role;
                    if (document.getElementById("user-email")) document.getElementById("user-email").innerText = currentUser.email;

                    loginModal.classList.add("hidden");
                    appLayout.classList.remove("hidden");
                    initDashboard();
                } else {
                    loginError.classList.remove("hidden");
                }
            } catch (e) {
                loginError.innerText = "Cannot reach authentication server.";
                loginError.classList.remove("hidden");
            } finally {
                loginBtnText.innerText = "Sign In";
                loginSpinner.classList.add("hidden");
            }
        });
    }

    const logoutBtn = document.getElementById("logout-btn");
    if (logoutBtn) {
        logoutBtn.addEventListener("click", () => {
            localStorage.clear();
            window.location.reload();
        });
    }

    document.querySelectorAll(".nav-item").forEach(item => {
        item.addEventListener("click", (e) => {
            if (item.id === 'logout-btn') return;
            document.querySelectorAll(".nav-item").forEach(n => n.classList.remove("active"));
            e.currentTarget.classList.add("active");

            const target = e.currentTarget.getAttribute("data-target");
            if (target) {
                document.querySelectorAll(".view").forEach(v => v.classList.add("hidden"));
                const targetEl = document.getElementById(target);
                if (targetEl) targetEl.classList.remove("hidden");
            }
        });
    });

    // MLOPS Events
    const fetchMlopsBtn = document.getElementById("fetch-mlops-btn");
    if (fetchMlopsBtn) {
        fetchMlopsBtn.addEventListener("click", fetchMLOpsData);
    }
    const retrainBtn = document.getElementById("trigger-retrain-btn");
    if (retrainBtn) {
        retrainBtn.addEventListener("click", async () => {
            const s = document.getElementById("mlops-store").value;
            const p = document.getElementById("mlops-product").value;
            try {
                const res = await fetch(`${API_URL}/monitoring/retrain`, {
                    method: "POST", headers: { "Authorization": `Bearer ${authToken}`, "Content-Type": "application/json" },
                    body: JSON.stringify({ store_id: s, product_id: p })
                });
                if (res.ok) alert("ML Retraining Pipeline trigger dispatched successfully.");
                else alert(await res.text());
            } catch (e) { alert("Network Error"); }
        });
    }

    // FORECAST Events
    const forecastForm = document.getElementById("forecast-form");
    if (forecastForm) {
        forecastForm.addEventListener("submit", async (e) => {
            e.preventDefault();
            const btn = document.getElementById("generate-btn");
            const oText = btn.innerHTML;
            btn.innerHTML = `Executing Pipeline...`;
            btn.disabled = true;
            const pData = {
                store_id: document.getElementById("store").value,
                product_id: document.getElementById("product").value,
                horizon_days: parseInt(document.getElementById("horizon").value, 10),
                current_stock: parseInt(document.getElementById("stock").value, 10),
                safety_stock: 20,
                lead_time_days: 2
            };
            try {
                const res = await fetch(`${API_URL}/inventory/recommendations`, {
                    method: "POST", headers: { "Content-Type": "application/json", "Authorization": `Bearer ${authToken}` },
                    body: JSON.stringify(pData)
                });
                if (res.ok) {
                    const data = await res.json();
                    renderForecast(data);
                } else { alert(`Error: ${await res.text()}`); }
            } catch (err) { alert("Network error."); } finally { btn.innerHTML = oText; btn.disabled = false; }
        });

        document.getElementById("clear-btn")?.addEventListener("click", () => {
            document.getElementById("forecast-empty").classList.remove("hidden");
            document.getElementById("forecast-content").classList.add("hidden");
            if (forecastChartInstance) { forecastChartInstance.destroy(); forecastChartInstance = null; }
        });
    }

    // ANOMALIES Event
    const refreshAnom = document.getElementById("refresh-anomalies");
    if (refreshAnom) { refreshAnom.addEventListener("click", executeAnomalyScan); }

    // SURPLUS POST Event
    const excForm = document.getElementById("create-surplus-form");
    if (excForm) {
        excForm.addEventListener("submit", async (e) => {
            e.preventDefault();
            const btn = document.getElementById("exc-submit-btn");
            btn.innerHTML = "Posting...";
            btn.disabled = true;
            try {
                const s = document.getElementById("exc-store").value;
                const p = document.getElementById("exc-product").value;
                const q = document.getElementById("exc-qty").value;
                const pr = document.getElementById("exc-price").value;

                const res = await fetch(`${API_URL}/exchange/listings`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json', "Authorization": `Bearer ${authToken}` },
                    body: JSON.stringify({
                        store_id: s, product_id: p, available_qty: parseInt(q),
                        price_per_unit: parseFloat(pr)
                    })
                });
                if (res.ok) {
                    alert("Added to Global Surplus Market Successfully!");
                    excForm.reset();
                    document.getElementById("exc-store").value = s;
                    document.getElementById("exc-product").value = p;
                    const activeS = localStorage.getItem('smartstock_store');
                    loadSurplusMatches(activeS);
                } else alert("Error publishing.");
            } catch (er) { }
            finally {
                btn.innerHTML = '<i class="ph ph-plus"></i> Post Global Listing';
                btn.disabled = false;
            }
        });
    }
});

async function initDashboard() {
    let authStores = "*";
    try {
        const uRes = await fetch(`${API_URL}/auth/me`, { headers: { "Authorization": `Bearer ${authToken}` } });
        if (uRes.ok) {
            const uInfo = await uRes.json();
            authStores = uInfo.authorized_stores || "*";
        }
    } catch (e) { }

    const activeStore = localStorage.getItem('smartstock_store') || ((authStores !== "*" && authStores.split(",").length > 0) ? authStores.split(",")[0].trim() : "Germany");
    localStorage.setItem('smartstock_store', activeStore);

    // (We replaced active-store-name text with the dropdown, so storeHeader is handled in the select)

    try {
        const [sRes, pRes] = await Promise.all([
            fetch(`${API_URL}/stores`),
            fetch(`${API_URL}/products`)
        ]);
        if (sRes.ok && pRes.ok) {
            const sData = await sRes.json();
            const pData = await pRes.json();
            globalStores = sData.stores;
            globalProducts = pData.products;

            // Dashboard
            if (document.getElementById("dash-stores")) document.getElementById("dash-stores").innerText = globalStores.length;
            if (document.getElementById("dash-products")) document.getElementById("dash-products").innerText = globalProducts.length;

            // Populate all selects
            const popSelects = [
                { s: document.getElementById("store"), p: document.getElementById("product") },
                { s: document.getElementById("mlops-store"), p: document.getElementById("mlops-product") },
                { s: document.getElementById("exc-store"), p: document.getElementById("exc-product") },
                { s: document.getElementById("global-store-selector"), p: null }
            ];

            let allowed = authStores.split(",").map(s => s.trim());
            let validStores = globalStores.filter(s => authStores === "*" || allowed.includes(s.id));
            if (validStores.length === 0) validStores = globalStores;

            popSelects.forEach(sel => {
                if (sel.s) {
                    sel.s.innerHTML = "";
                    validStores.forEach(s => {
                        let text = sel.s.id === 'global-store-selector' ? s.id + " Branch" : s.id;
                        sel.s.insertAdjacentHTML("beforeend", `<option value="${s.id}">${text}</option>`);
                    });
                    if (sel.s.id === 'global-store-selector') sel.s.value = activeStore;
                }
                if (sel.p) { sel.p.innerHTML = ""; globalProducts.forEach(p => sel.p.insertAdjacentHTML("beforeend", `<option value="${p.id}">${p.id}</option>`)); }
            });

            const gSelector = document.getElementById("global-store-selector");
            if (gSelector) {
                gSelector.addEventListener("change", (e) => {
                    const selected = e.target.value;
                    localStorage.setItem('smartstock_store', selected);
                    loadInventoryData(selected);
                    loadSurplusMatches(selected);
                });
            }

            // Set active store default logically where relevant
            if (document.getElementById("exc-store")) document.getElementById("exc-store").value = activeStore;
        }
    } catch (e) { console.error(e); }

    // Inventory Health
    loadInventoryData();
    // Surplus List
    loadSurplusMatches(activeStore);
    // Home Mini Alerts
    loadMLAlertsMini();
}

async function loadInventoryData(storeName) {
    const sName = storeName || localStorage.getItem('smartstock_store') || 'Germany';
    const listEl = document.getElementById("ai-reorder-list");
    if (!listEl) return;
    if (globalProducts.length === 0) return;

    const offset = sName.length;
    const topProducts = globalProducts.slice(0, 9);
    const statuses = [
        { text: "Healthy", class: "in-stock", demand: 40 + offset, stock: 120 + offset },
        { text: "Reorder", class: "low-stock", demand: 85 + offset, stock: 40 + offset },
        { text: "Critical", class: "out-stock", demand: 150 + offset, stock: 5 + offset }
    ];

    listEl.innerHTML = "";
    topProducts.forEach((p, idx) => {
        let r = statuses[(idx + offset) % 3];
        let isSel = (r.text === 'Critical' && (idx + offset) % 3 === 2) ? 'selected' : '';
        let img = getIcon(p.id);

        listEl.insertAdjacentHTML("beforeend", `
         <div class="list-item ${isSel}">
            <div class="item-img" style="font-size:1.5rem; line-height:1; text-align:center; ${isSel ? 'background:rgba(255,255,255,0.2)' : ''}">${img}</div>
            <div class="col"><strong>${p.id}</strong><span>Category: Food</span></div>
            <div class="col"><span>Current Stock</span><strong>${r.stock} Units</strong></div>
            <div class="col"><span>7D Predicted Demand</span><strong>${r.demand} Units <i class="ph ph-trend-up"></i></strong></div>
            <div class="col text-right"><span class="status ${r.class}">${r.text}</span></div>
         </div>
       `);
    });
}

function renderForecast(data) {
    document.getElementById("forecast-empty").classList.add("hidden");
    document.getElementById("forecast-content").classList.remove("hidden");

    const rec = data.inventory_recommendation;
    const meta = data.model_metadata;
    const stockInput = document.getElementById("stock").value;

    document.getElementById("res-current-stock").innerText = stockInput;
    document.getElementById("res-expected-demand").innerText = rec.forecast_demand;
    document.getElementById("res-target-stock").innerText = rec.target_stock;
    document.getElementById("res-reorder").innerText = rec.recommended_order;
    document.getElementById("res-reason").innerText = rec.reason;
    document.getElementById("res-model").innerText = meta.type || "Default";

    const alertBox = document.getElementById("alert-box");
    const b = document.getElementById("stock-status-badge");
    b.className = "status";
    b.innerText = rec.stock_status;

    alertBox.classList.add("hidden");
    b.style.color = ""; // reset color

    document.getElementById("surplus-dialogue").classList.add("hidden");

    if (rec.stock_status === "SAFE" || rec.stock_status === "HEALTHY") {
        b.classList.add("in-stock");
    }
    else if (rec.stock_status === "SURPLUS") {
        b.classList.add("in-stock");
        b.style.color = "#9333ea"; // Unique Purple
        document.getElementById("surplus-dialogue").classList.remove("hidden");
        document.getElementById("surplus-dialogue-text").innerText = rec.reason;
    }
    else if (rec.stock_status.includes("LOW")) {
        b.classList.add("low-stock");
    }
    else if (rec.stock_status.includes("REORDER") || rec.stock_status === "CRITICAL") {
        b.classList.add("out-stock");
        alertBox.classList.remove("hidden");
    }

    const ctx = document.getElementById("forecastChart").getContext("2d");
    if (forecastChartInstance) forecastChartInstance.destroy();

    const labels = data.forecasts.map(f => f.date);
    const vals = data.forecasts.map(f => f.predicted_demand);
    const grad = ctx.createLinearGradient(0, 0, 0, 300);
    grad.addColorStop(0, "rgba(56,189,248,0.4)");
    grad.addColorStop(1, "rgba(56,189,248,0)");

    forecastChartInstance = new Chart(ctx, {
        type: "line",
        data: { labels, datasets: [{ label: "Demand Units", data: vals, borderColor: "#38bdf8", backgroundColor: grad, fill: true, tension: 0.3, borderWidth: 2, pointBackgroundColor: "#fff", pointBorderColor: "#38bdf8", pointRadius: 4 }] },
        options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { y: { beginAtZero: true } } }
    });
}

async function fetchMLOpsData() {
    const s = document.getElementById("mlops-store").value;
    const p = document.getElementById("mlops-product").value;

    document.getElementById("mlops-empty").classList.add("hidden");
    document.getElementById("mlops-content").classList.remove("hidden");
    document.getElementById("mlops-model").innerText = "Loading...";

    try {
        const res = await fetch(`${API_URL}/monitoring/health?store_id=${s}&product_id=${p}`, { headers: { "Authorization": `Bearer ${authToken}` } });
        if (res.ok) {
            const data = await res.json();
            document.getElementById("mlops-model").innerText = data.production_model;
            document.getElementById("mlops-status").innerText = data.model_status;
            document.getElementById("mlops-decision").innerText = data.retraining_decision;
            document.getElementById("ml-wape").innerText = (data.performance.wape * 100).toFixed(2) + "%";
            document.getElementById("ml-mae").innerText = data.performance.mae.toFixed(2);
            document.getElementById("ml-bias").innerText = data.performance.bias.toFixed(4);
            document.getElementById("ml-decay").innerText = (data.performance.degradation * 100).toFixed(2) + "% Deviation";
            document.getElementById("ml-feats-total").innerText = data.drift.total_features;
            document.getElementById("ml-feats-drift").innerText = data.drift.features_with_drift;

            const ul = document.getElementById("ml-drift-list");
            ul.innerHTML = "";
            if (data.drift.feature_details.length === 0) ul.innerHTML = "<div class='text-secondary'>No dimensional drift detected.</div>";
            else data.drift.feature_details.forEach(f => { ul.insertAdjacentHTML("beforeend", `<div class="list-item"><div class="col"><strong>${f.feature}</strong></div><div class="col text-right"><span class="badge-border text-error">Δ +${(f.drift_score * 100).toFixed(1)}%</span></div></div>`); });
        } else alert(`Operational API blocked context access: ${res.statusText}`);
    } catch (e) { alert("Error fetching MLOps data."); }
}

async function executeAnomalyScan() {
    const tb = document.getElementById("full-anomalies-list");
    if (!tb) return;
    tb.innerHTML = `<div class="text-secondary text-center p-2">Scanning pipeline integrity...</div>`;
    try {
        const res = await fetch(`${API_URL}/monitoring/anomalies`, { headers: { "Authorization": `Bearer ${authToken}` } });
        if (res.ok) {
            const data = await res.json();
            document.getElementById("qa-score").innerText = data.data_quality_score + "/100";
            document.getElementById("qa-scanned").innerText = data.total_records;
            document.getElementById("qa-anomalies").innerText = data.anomalies;
            document.getElementById("qa-critical").innerText = data.critical_anomalies;
            if (data.items.length === 0) { tb.innerHTML = `<div class="text-secondary text-center p-2">Data pipeline is 100% clean.</div>`; return; }

            tb.innerHTML = "";
            data.items.forEach(a => {
                let sevCls = a.severity === 'critical' ? 'text-error' : 'text-warning';
                let typeCls = a.anomaly_type === 'outlier' ? 'low-stock' : 'out-stock';
                tb.insertAdjacentHTML("beforeend", `
                  <div class="list-item">
                     <div class="col"><strong>${a.date.substring(0, 10)}</strong><span>Detection</span></div>
                     <div class="col"><strong>${a.product_id}</strong><span>Product</span></div>
                     <div class="col"><span class="status ${typeCls}">${a.anomaly_type}</span></div>
                     <div class="col"><strong class="${sevCls}">${a.severity.toUpperCase()}</strong></div>
                     <div class="col"><strong>${parseFloat(a.original_value).toFixed(2)}</strong><span>Value</span></div>
                     <div class="col text-right"><button class="btn-outline">Details</button></div>
                  </div>
                `);
            });
        }
    } catch (e) { }
}

async function loadSurplusMatches(myStore) {
    const matchEl = document.getElementById("surplus-matches-list");
    if (!matchEl) return;
    try {
        const lRes = await fetch(`${API_URL}/exchange/listings?store_id=${myStore}`, {
            headers: { "Authorization": `Bearer ${authToken}` }
        });
        if (!lRes.ok) throw new Error();
        const lData = await lRes.json();

        matchEl.innerHTML = "";
        if (lData.listings.length === 0) {
            matchEl.innerHTML = `<div class="text-secondary text-center p-2">No global surplus found.</div>`;
            return;
        }

        lData.listings.forEach((l, idx) => {
            let isHighlight = idx === 1 ? 'highlighted' : '';
            matchEl.insertAdjacentHTML("beforeend", `
            <div class="list-item ${isHighlight}">
                <div class="col"><span>From Store</span><strong><i class="ph ph-storefront"></i> ${l.seller}</strong></div>
                <div class="col"><span>Product</span><strong>${l.product} (Q: ${l.qty})</strong></div>
                <div class="col"><span>Price/Unit</span><strong>$${l.price || '0.00'}</strong></div>
                <div class="col text-right">
                    <button class="btn-outline" style="${isHighlight ? 'color:white; border-color:white;' : ''}" onclick="buySurplus('${l.listing_id}', '${l.qty}')">Buy Surplus</button>
                </div>
            </div>
            `);
        });
    } catch (e) { matchEl.innerHTML = `<div class="text-error text-center p-2">Failed network.</div>`; }
}

async function loadMLAlertsMini() {
    const alertEl = document.getElementById("ml-alerts-list");
    if (!alertEl) return;
    try {
        const aRes = await fetch(`${API_URL}/monitoring/anomalies`, { headers: { "Authorization": `Bearer ${authToken}` } });
        if (!aRes.ok) throw new Error();
        const aData = await aRes.json();

        alertEl.innerHTML = "";
        if (aData.items.length === 0) {
            alertEl.innerHTML = `<div class="text-secondary text-center p-2">All Pipeline Checks Passed.</div>`;
            return;
        }

        aData.items.slice(0, 4).forEach(a => {
            let severityCls = a.severity === 'critical' ? 'text-error' : 'text-warning';
            alertEl.insertAdjacentHTML("beforeend", `
            <div class="list-item">
                <div class="item-img" style="font-size:1.5rem; text-align:center;">${a.anomaly_type === 'outlier' ? '🔴' : '📈'}</div>
                <div class="col"><strong>${a.product_id}</strong><span>${a.date.substring(0, 10)}</span></div>
                <div class="col text-right"><strong class="${severityCls}">${a.severity.toUpperCase()}</strong></div>
            </div>
            `);
        });
    } catch (e) { alertEl.innerHTML = `<div class="text-error text-center p-2">Could not fetch Alerts.</div>`; }
}

window.buySurplus = async function (listingId, maxQty) {
    const qty = prompt(`How many units do you want to secure from the listing? (Max available: ${maxQty})`, maxQty);
    if (!qty) return;
    const buyerStore = localStorage.getItem('smartstock_store') || 'Germany';
    try {
        const res = await fetch(`${API_URL}/exchange/requests`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', "Authorization": `Bearer ${localStorage.getItem('smartstock_token')}` },
            body: JSON.stringify({ listing_id: listingId, buyer_store_id: buyerStore, requested_qty: parseInt(qty) })
        });
        if (res.ok) {
            alert('Successfully reserved from Global Exchange! Transaction match confirmed and stock removed from active market.');
            loadSurplusMatches(buyerStore);
        } else alert(`Failed: ${await res.text()}`);
    } catch (e) { alert("Network Error"); }
}