document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('screening-form');
    const resultContainer = document.getElementById('result-container');
    const submitBtn = document.getElementById('submit-btn');
    const btnText = submitBtn.querySelector('.btn-text');
    const btnLoader = submitBtn.querySelector('.btn-loader');
    const closeBtn = document.getElementById('close-result');
    
    const progressBar = document.getElementById('progress-bar');
    const probText = document.getElementById('prob-text');
    const predictionText = document.getElementById('prediction-text');
    const analysisMessage = document.getElementById('analysis-message');
    const resultCard = document.querySelector('.result-card');

    let shapChartInstance = null; // Global chart reference
    
    const xaiContainer = document.getElementById('xai-container');
    const downloadBtn = document.getElementById('download-report');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // Show loading state
        btnText.classList.add('hidden');
        btnLoader.classList.remove('hidden');
        submitBtn.disabled = true;

        const formData = new FormData(form);
        const data = {};
        
        formData.forEach((value, key) => {
            // Convert numerical strings to numbers
            if (key.startsWith('a') && key.length <= 3) {
                data[key] = parseInt(value);
                // Save string for report 
                data[`${key}_str`] = value === "1" ? "Trait Present (Risk)" : "Typical Behavior";
            } else {
                data[key] = value;
            }
        });

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            });

            const result = await response.json();

            if (result.success) {
                showResult(result, data);
            } else {
                alert('Error: ' + result.error);
            }
        } catch (error) {
            console.error('Fetch Error:', error);
            alert('Could not connect to the ML model server. Please ensure the backend is running.');
        } finally {
            // Reset button state
            btnText.classList.remove('hidden');
            btnLoader.classList.add('hidden');
            submitBtn.disabled = false;
        }
    });

    closeBtn.addEventListener('click', () => {
        resultContainer.classList.add('hidden');
        // Reset progress bar for next time
        progressBar.style.strokeDashoffset = '283';
    });

    downloadBtn.addEventListener('click', () => {
        const element = document.getElementById('medical-report-template');
        
        // Extract the raw HTML string and force it to display: block
        // By passing a raw string, html2pdf creates a clean, isolated iframe to render the PDF
        // entirely bypassing any viewport, z-index, or CSS layout bugs happening on the live screen.
        let htmlContent = element.outerHTML;
        htmlContent = htmlContent.replace('display: none;', 'display: block;');
        
        const opt = {
            margin:       10,
            filename:     'AuraScan_Clinical_Report.pdf',
            image:        { type: 'jpeg', quality: 1.0 },
            html2canvas:  { scale: 2, useCORS: true, logging: false },
            jsPDF:        { unit: 'mm', format: 'a4', orientation: 'portrait' }
        };
        
        const originalText = downloadBtn.innerHTML;
        downloadBtn.innerHTML = 'Generating PDF...';
        downloadBtn.disabled = true;
        
        html2pdf().set(opt).from(htmlContent).save().then(() => {
            downloadBtn.innerHTML = originalText;
            downloadBtn.disabled = false;
        }).catch(err => {
            console.error("PDF Generation Error:", err);
            alert("There was an issue generating the report. Please try again.");
            downloadBtn.innerHTML = originalText;
            downloadBtn.disabled = false;
        });
    });

    function showResult(resultData, formData) {
        const prob = resultData.probability * 100;
        const isPositive = resultData.prediction.toUpperCase() === 'YES';
        
        // Update text
        predictionText.textContent = isPositive ? 'ASD Characteristics Detected' : 'No ASD Characteristics Detected';
        predictionText.style.color = isPositive ? '#ef4444' : '#22c55e';
        
        analysisMessage.textContent = isPositive 
            ? 'The model indicates a high probability of Autism Spectrum traits. We recommend consulting a healthcare professional for a detailed evaluation.'
            : 'The model indicates a low probability of Autism Spectrum traits. However, if you have concerns, professional consultation is always advised.';

        probText.textContent = `${Math.round(prob)}%`;
        
        // Update Explainable AI factors if present
        if (resultData.factors && resultData.factors.length > 0) {
            // For the medical report
            const repFactors = document.getElementById('rep-factors');
            if (repFactors) repFactors.innerHTML = '';
            
            const labels = [];
            const dataImpacts = [];
            const backgroundColors = [];
            
            resultData.factors.forEach(factorObj => {
                labels.push(factorObj.feature);
                dataImpacts.push(Math.abs(factorObj.impact));
                
                // Red if pushing towards Autism, Green if pushing away, neutral if small
                const isASD = resultData.prediction.toUpperCase() === 'YES';
                backgroundColors.push(isASD ? '#ef4444' : '#10b981');
                
                // Report update
                if (repFactors) {
                    const repLi = document.createElement('li');
                    repLi.innerHTML = `<strong>${factorObj.feature}</strong>: Impact value ${Math.abs(factorObj.impact).toFixed(4)}`;
                    repFactors.appendChild(repLi);
                }
            });
            
            xaiContainer.classList.remove('hidden');
            
            // Draw Chart
            const ctx = document.getElementById('shapChart').getContext('2d');
            if (shapChartInstance) {
                shapChartInstance.destroy(); // Clear old chart
            }
            
            shapChartInstance = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Feature Impact Magnitude',
                        data: dataImpacts,
                        backgroundColor: backgroundColors,
                        borderRadius: 4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    indexAxis: 'y', // Make it horizontal
                    plugins: {
                        legend: { display: false }
                    },
                    scales: {
                        x: {
                            grid: { color: 'rgba(255, 255, 255, 0.05)' },
                            ticks: { color: '#94a3b8' }
                        },
                        y: {
                            grid: { display: false },
                            ticks: { color: '#f8fafc', font: { family: 'Inter', size: 11 } }
                        }
                    }
                }
            });
            
        } else {
            xaiContainer.classList.add('hidden');
            if (document.getElementById('rep-factors')) {
                document.getElementById('rep-factors').innerHTML = '<li>No specific dominant factors identified outside normal ranges.</li>';
            }
        }


        // Populate Medical Report Template
        document.getElementById('report-date').textContent = new Date().toLocaleDateString();
        document.getElementById('rep-sex').textContent = formData.sex === 'm' ? 'Male' : 'Female';
        document.getElementById('rep-jaundice').textContent = formData.jaundice === 'yes' ? 'Yes' : 'No';
        document.getElementById('rep-family').textContent = formData.family_asd === 'yes' ? 'Positive (Yes)' : 'None reported (No)';
        document.getElementById('rep-prediction').textContent = isPositive ? 'ASD Characteristics Detected' : 'No ASD Characteristics Detected';
        document.getElementById('rep-prediction').style.color = isPositive ? '#ef4444' : '#10b981';
        document.getElementById('rep-confidence').textContent = `${Math.round(prob)}%`;
        
        // Populate Behaviors Table
        const obsTable = document.getElementById('rep-behavioral');
        obsTable.innerHTML = '';
        const behavioralLabels = [
            "A1: Social Interaction", "A2: Shared Attention", "A3: Social Communication",
            "A4: Imagination", "A5: Repetitive Behavior", "A6: Social Interest",
            "A7: Attention to Detail", "A8: Communication Development", "A9: Play & Social Interaction", "A10: Understanding Others"
        ];
        
        for (let i = 1; i <= 10; i++) {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td style="padding: 8px; border: 1px solid #cbd5e1;">${behavioralLabels[i-1]}</td>
                <td style="padding: 8px; border: 1px solid #cbd5e1;">${formData[`a${i}_str`]}</td>
            `;
            obsTable.appendChild(tr);
        }

        // Show download button
        downloadBtn.classList.remove('hidden');
        
        // Update circle progress
        const offset = 283 - (prob / 100) * 283;
        progressBar.style.strokeDashoffset = offset;
        
        // Add status class to card
        resultCard.classList.remove('positive', 'negative');
        resultCard.classList.add(isPositive ? 'positive' : 'negative');
        
        // Show container
        resultContainer.classList.remove('hidden');
    }
});
