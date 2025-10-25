// Wait for the page to load before running scripts
document.addEventListener("DOMContentLoaded", () => {
  // --- 1. Get all the HTML elements we need ---
  const startupForm = document.getElementById("startup-form");
  const userIdeaInput = document.getElementById("user-idea");
  const userCityInput = document.getElementById("user-city");
  const foundingYearInput = document.getElementById("founding-year");
  const analyzeButton = document.getElementById("analyze-button");
  
  const spinner = document.getElementById("spinner");
  const errorBox = document.getElementById("error-box");
  const resultsSection = document.getElementById("results-section");

  // Get result containers
  const profileSection = document.getElementById("profile-section");
  const aiAnalysisSection = document.getElementById("ai-analysis-section");
  const similarStartupsSection = document.getElementById("similar-startups-section");
  const fundingAnalysisSection = document.getElementById("funding-analysis-section");
  const extraPredictionSection = document.getElementById("extra-prediction-section");

  // --- 2. Set default founding year ---
  foundingYearInput.value = new Date().getFullYear();
  foundingYearInput.max = new Date().getFullYear() + 5;
  foundingYearInput.min = new Date().getFullYear() - 10;

  // --- 3. Listen for the form submission ---
  startupForm.addEventListener("submit", async (e) => {
    e.preventDefault(); // Stop the form from reloading the page

    // Get values from the form
    const userIdea = userIdeaInput.value;
    const userCity = userCityInput.value;
    const foundingYear = parseInt(foundingYearInput.value, 10);

    // --- 4. Simple frontend validation ---
    if (!userIdea || !userCity || !foundingYear) {
      showError("Please fill out all fields: Idea, City, and Year.");
      return;
    }

    // --- 5. Prepare the UI for loading ---
    showLoading(true);
    
    // --- 6. Send data to the FastAPI backend ---
    try {
      const response = await fetch("http://127.0.0.1:8000/analyze", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          user_idea: userIdea,
          user_city: userCity,
          founding_year: foundingYear,
        }),
      });

      // Handle bad responses (like 503, 500, etc.)
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "An unknown error occurred.");
      }

      // We got a good response
      const data = await response.json();
      populateResults(data);

    } catch (error) {
      // Handle network errors or errors from the `throw` above
      console.error("Fetch Error:", error);
      showError(error.message);
    } finally {
      // --- 8. Always hide the spinner when done ---
      showLoading(false);
    }
  });

  // --- Helper function to show/hide loading state ---
  function showLoading(isLoading) {
    if (isLoading) {
      spinner.classList.remove("hidden");
      analyzeButton.disabled = true;
      analyzeButton.textContent = "Analyzing...";
      errorBox.classList.add("hidden");
      resultsSection.classList.add("hidden");
    } else {
      spinner.classList.add("hidden");
      analyzeButton.disabled = false;
      analyzeButton.textContent = "Analyze My Idea";
    }
  }

  // --- Helper function to show an error message ---
  function showError(message) {
    errorBox.textContent = message;
    errorBox.classList.remove("hidden");
    resultsSection.classList.add("hidden");
  }

  // --- Helper function to build all the result HTML ---
  function populateResults(data) {
    // 1. Profile
    const profile = data.profile;
    profileSection.innerHTML = `
      <div class="metric">
        <span class="metric-label">Predicted Industry</span>
        <span class="metric-value">${profile.predicted_industry}</span>
      </div>
      <div class="metric">
        <span class="metric-label">Your City</span>
        <span class="metric-value">${profile.user_city}</span>
      </div>
      <div class="metric">
        <span class="metric-label">Founding Year</span>
        <span class="metric-value">${profile.founding_year}</span>
      </div>
    `;

    // 2. AI Analysis
    const ai = data.ai_analysis;
    const verdictClass = ai.is_copy ? 'ai-verdict-copy' : 'ai-verdict-unique';
    const verdictText = ai.is_copy ? "High 'Copycat' Risk" : "Looks Unique!";
    
    aiAnalysisSection.innerHTML = `
      <strong>Similarity to closest match (${ai.top_match_name}):</strong>
      <div class="progress-bar-container">
        <div class="progress-bar" style="width: ${ai.score}%;">
          ${ai.score}%
        </div>
      </div>
      <div class="ai-reasoning ${verdictClass}">
        <strong>AI Verdict: ${verdictText}</strong><br>
        ${ai.reasoning}
      </div>
    `;

    // 3. Similar Startups
    similarStartupsSection.innerHTML = data.similar_startups.map((startup, index) => {
      // Format funding to currency
      const funding = new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        maximumFractionDigits: 0
      }).format(startup['Funding Amount in $']);

      return `
        <div class="startup-card">
          <h3 class="startup-card-header">${index + 1}. ${startup.Company}</h3>
          <div class="startup-card-subheader">Industry: ${startup.Industries}</div>
          <p class="startup-card-description">${startup.Description}</p>
          <div class="startup-card-details">
            <strong>Funding:</strong> ${funding} (Round: ${startup['Funding Round']})<br>
            <strong>Location:</strong> ${startup.City}
          </div>
        </div>
      `;
    }).join(''); // Join all the HTML strings together

    // 4. Funding Analysis
    const funding = data.funding_analysis;
    fundingAnalysisSection.innerHTML = `
      <div class="metric">
        <span class="metric-label">Funding Rate (in sample)</span>
        <span class="metric-value">${funding.rate}</span>
        <span class="metric-delta">startups were funded</span>
      </div>
      <div class="metric">
        <span class="metric-label">Average Funding (for funded)</span>
        <span class="metric-value">${funding.avg_funding_str}</span>
      </div>
    `;

    // 5. Extra Prediction (Your Joblib placeholder)
    extraPredictionSection.innerHTML = `
      <div class="metric">
        <span class="metric-label">Custom Model Prediction</span>
        <span class="metric-value">${data.extra_prediction}</span>
      </div>
    `;

    // Finally, show the whole results section
    resultsSection.classList.remove("hidden");
  }
});