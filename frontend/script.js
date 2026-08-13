const API_URL = "http://localhost:8000";
let chartInstance = null;

document.addEventListener("DOMContentLoaded", async () => {
    // DOM Elements
    const storeSelect = document.getElementById("store");
    const productSelect = document.getElementById("product");
    const form = document.getElementById("forecast-form");
    const generateBtn = document.getElementById("generate-btn");
    const spinner = document.getElementById("loading-spinner");
    const resultsContainer = document.getElementById("results-container");
    const alertBox = document.getElementById("alert-box");

    // Fetch initial data
    try {
        const [storesRes, productsRes] = await Promise.all([
            fetch(`${API_URL}/stores`),
            fetch(`${API_URL}/products`)
        ]);

        const storesData = await storesRes.json();
        const productsData = await productsRes.json();

        document.getElementById("total-products").innerText = productsData.products.length;

        storesData.stores.forEach(s => {
            const opt = document.createElement("option");
            opt.value = s.id;
            opt.textContent = s.id;
            storeSelect.appendChild(opt);
        });

        // P001, P002 ... P020 logic directly (fallback mapping because our synthetic data uses exact P0XX vs mock strings)
        // Ensure products populate cleanly for the 20 products
        productsData.products.forEach(p => {
            const opt = document.createElement("option");
            opt.value = p.id;
            opt.textContent = p.name ? `${p.id} - ${p.name}` : p.id;
            productSelect.appendChild(opt);
        });

        // Dynamic rendering works fine. Removed synthetic fallback.

    } catch (error) {
        console.error("Error fetching initial data:", error);
    }

    form.addEventListener("submit", async (e) => {
        e.preventDefault();

        const storeId = storeSelect.value;
        const productId = productSelect.value;
        const horizon = parseInt(document.getElementById("horizon").value, 10);
        const stock = parseInt(document.getElementById("stock").value, 10);

        // UI states
        generateBtn.querySelector("span").innerText = "Generating...";
        spinner.classList.remove("hidden");
        generateBtn.disabled = true;
        alertBox.classList.add("hidden");
        resultsContainer.classList.add("hidden");

        // Prepare request body
        const requestBody = {
            store_id: storeId,
            product_id: productId,
            horizon_days: horizon,
            current_stock: stock,
            safety_stock: 20, // using default
            lead_time_days: 2
        };

        try {
            const res = await fetch(`${API_URL}/forecast`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(requestBody)
            });

            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.detail || "Failed to fetch");
            }

            const data = await res.json();
            updateDashboard(data.inventory_recommendation, stock);
            renderChart(data.forecasts);
            resultsContainer.classList.remove("hidden");

        } catch (error) {
            alert(`Error: ${error.message}`);
        } finally {
            generateBtn.querySelector("span").innerText = "Generate Forecast";
            spinner.classList.add("hidden");
            generateBtn.disabled = false;
        }
    });

    function updateDashboard(recommendation, currentStock) {
        document.getElementById("res-current-stock").innerText = currentStock;
        document.getElementById("res-expected-demand").innerText = recommendation.total_forecast_demand;
        document.getElementById("res-safety-stock").innerText = 20; // default used above
        document.getElementById("res-reorder").innerText = recommendation.recommended_reorder;

        const badge = document.getElementById("stock-status-badge");
        let statusText = recommendation.status;

        badge.innerText = statusText;
        badge.className = "badge"; // reset

        // We mapped status styles in CSS via classes
        if (statusText === "SAFE") badge.classList.add("SAFE");
        else if (statusText === "LOW STOCK") {
            badge.classList.add("LOW");
            badge.innerText = "LOW";
        }
        else if (statusText === "REORDER NOW") {
            badge.classList.add("REORDER");
            badge.innerText = "REORDER";
            alertBox.classList.remove("hidden");
        }
        else if (statusText === "CRITICAL") {
            badge.classList.add("CRITICAL");
            badge.innerText = "CRITICAL";
            alertBox.classList.remove("hidden");
        }
    }

    function renderChart(forecasts) {
        const ctx = document.getElementById('forecastChart').getContext('2d');

        const labels = forecasts.map(f => f.date);
        const dataPts = forecasts.map(f => f.predicted_demand);

        if (chartInstance) {
            chartInstance.destroy();
        }

        Chart.defaults.color = "#4b5563";
        Chart.defaults.font.family = "'Inter', sans-serif";

        const gradient = ctx.createLinearGradient(0, 0, 0, 400);
        gradient.addColorStop(0, 'rgba(59, 130, 246, 0.5)');
        gradient.addColorStop(1, 'rgba(59, 130, 246, 0.0)');

        chartInstance = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Predicted Units Sold',
                    data: dataPts,
                    borderColor: '#3b82f6',
                    backgroundColor: gradient,
                    borderWidth: 3,
                    pointBackgroundColor: '#fff',
                    pointBorderColor: '#3b82f6',
                    pointBorderWidth: 2,
                    pointRadius: 4,
                    pointHoverRadius: 6,
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        backgroundColor: 'rgba(17, 24, 39, 0.9)',
                        titleFont: { size: 14 },
                        bodyFont: { size: 14 },
                        padding: 12,
                        cornerRadius: 6,
                        displayColors: false
                    }
                },
                scales: {
                    x: {
                        grid: { color: 'rgba(0, 0, 0, 0.05)' }
                    },
                    y: {
                        beginAtZero: true,
                        grid: { color: 'rgba(0, 0, 0, 0.05)' }
                    }
                }
            }
        });
    }
});
