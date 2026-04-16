document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('prediction-form');
    const submitBtn = document.getElementById('submit-btn');
    const btnText = document.getElementById('btn-text');
    const btnLoader = document.getElementById('btn-loader');
    const resultContainer = document.getElementById('result-container');
    const predictedValue = document.getElementById('predicted-value');
    const riskCategory = document.getElementById('risk-category');
    const confidenceInterval = document.getElementById('confidence-interval');
    const errorToast = document.getElementById('error-toast');
    const toastMsg = document.getElementById('toast-msg');

    let toastTimeout;

    const showToast = (message) => {
        toastMsg.textContent = message;
        errorToast.classList.remove('hidden');
        // Small delay to allow display flex to apply before opacity transition
        setTimeout(() => errorToast.classList.add('show'), 10);
        
        clearTimeout(toastTimeout);
        toastTimeout = setTimeout(() => {
            errorToast.classList.remove('show');
            setTimeout(() => errorToast.classList.add('hidden'), 300);
        }, 4000);
    };

    const setLoading = (isLoading) => {
        if (isLoading) {
            btnText.classList.add('hidden');
            btnLoader.classList.remove('hidden');
            submitBtn.style.pointerEvents = 'none';
            submitBtn.style.opacity = '0.8';
        } else {
            btnText.classList.remove('hidden');
            btnLoader.classList.add('hidden');
            submitBtn.style.pointerEvents = 'auto';
            submitBtn.style.opacity = '1';
        }
    };

    const animateValue = (obj, start, end, duration) => {
        let startTimestamp = null;
        const step = (timestamp) => {
            if (!startTimestamp) startTimestamp = timestamp;
            const progress = Math.min((timestamp - startTimestamp) / duration, 1);
            // Ease out cubic
            const easeOut = 1 - Math.pow(1 - progress, 3);
            const current = (progress * (end - start) + start).toFixed(2);
            obj.innerHTML = (end > 0 ? '+' : '') + current + '%';
            
            if (current < 0) {
                obj.classList.add('negative');
            } else {
                obj.classList.remove('negative');
            }

            if (progress < 1) {
                window.requestAnimationFrame(step);
            } else {
                obj.innerHTML = (end > 0 ? '+' : '') + end.toFixed(2) + '%';
            }
        };
        window.requestAnimationFrame(step);
    };

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const payload = {
            company_name: document.getElementById('company_name').value,
            size_of_ipo: document.getElementById('size_of_ipo').value,
            pe_ratio: document.getElementById('pe_ratio').value,
            subscription_status: document.getElementById('subscription_status').value
        };

        setLoading(true);
        resultContainer.classList.add('hidden');

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(payload)
            });

            const data = await response.json();

            if (!response.ok || data.error) {
                throw new Error(data.error || 'Prediction failed.');
            }

            // Populate Results
            resultContainer.classList.remove('hidden');
            riskCategory.textContent = data.risk_category;
            
            const [ci_low, ci_high] = data.confidence_interval;
            confidenceInterval.textContent = `${ci_low.toFixed(2)}% to ${ci_high.toFixed(2)}%`;

            // Animate number from 0 to actual value
            animateValue(predictedValue, 0, data.predicted_gain, 1500);

            // Populate Market Pulse
            if (data.market_pulse) {
                const pulse = data.market_pulse;
                const sentimentTag = document.getElementById('sentiment-tag');
                const gmpVal = document.getElementById('gmp-val');
                const aiSignal = document.getElementById('ai-signal');
                const newsList = document.getElementById('news-list');

                // Sentiment Badge
                sentimentTag.textContent = pulse.sentiment_label;
                sentimentTag.className = 'sentiment-badge'; // reset
                if (pulse.sentiment_label.toLowerCase().includes('bullish')) sentimentTag.classList.add('bullish');
                if (pulse.sentiment_label.toLowerCase().includes('bearish')) sentimentTag.classList.add('bearish');

                // Stats
                gmpVal.textContent = pulse.chittorgarh_gmp || (pulse.gmp_estimate ? `₹${pulse.gmp_estimate}` : '--');
                const signal = pulse.avg_sentiment > 0 ? `+${(pulse.avg_sentiment * 100).toFixed(0)}% Boost` : `${(pulse.avg_sentiment * 100).toFixed(0)}% Drag`;
                aiSignal.textContent = signal;

                // News
                newsList.innerHTML = '';
                if (pulse.news && pulse.news.length > 0) {
                    pulse.news.slice(0, 3).forEach(item => {
                        const newsEl = document.createElement('a');
                        newsEl.href = item.link;
                        newsEl.target = '_blank';
                        newsEl.className = 'news-item';
                        newsEl.innerHTML = `
                            <div class="news-title">${item.title}</div>
                            <div class="news-meta">Sentiment: ${item.sentiment > 0 ? 'Bullish' : 'Neutral'}</div>
                        `;
                        newsList.appendChild(newsEl);
                    });
                } else {
                    newsList.innerHTML = '<div class="news-placeholder">No recent news found.</div>';
                }
            }

            // Scroll down a bit on mobile
            if(window.innerWidth < 1024) {
                setTimeout(()=> resultContainer.scrollIntoView({behavior: "smooth", block: 'nearest'}), 100);
            }

            // Populate Sector Info
            if (data.sector_analysis) {
                const sectorRank = document.getElementById('sector-rank');
                sectorRank.textContent = data.sector_analysis.performance;
                // Add color classes if needed
                sectorRank.className = 'metric-val';
                if (data.sector_analysis.performance === 'Outperformer') sectorRank.style.color = 'var(--success)';
                if (data.sector_analysis.performance === 'Underperformer') sectorRank.style.color = 'var(--danger)';
            }
        } catch (error) {
            showToast(error.message);
        } finally {
            setLoading(false);
        }
    });

    // Insights Toggle Logic
    const toggleInsightsBtn = document.getElementById('toggle-insights');
    const closeInsightsBtn = document.getElementById('close-insights');
    const insightsSection = document.getElementById('insights-section');

    const toggleInsights = (show) => {
        if (show) {
            insightsSection.classList.remove('hidden');
            document.body.style.overflow = 'hidden'; // Prevent scroll
        } else {
            insightsSection.classList.add('hidden');
            document.body.style.overflow = 'auto';
        }
    };

    toggleInsightsBtn.addEventListener('click', () => toggleInsights(true));
    closeInsightsBtn.addEventListener('click', () => toggleInsights(false));

    // Close on escape key
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') toggleInsights(false);
    });

    // Handle initial Lucide icons 
    // They are instantiated by the inline script on the HTML.
});
