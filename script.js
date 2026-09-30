lucide.createIcons();

const form = document.getElementById('predict-form');
const submitBtn = document.getElementById('submit-btn');
const resultEmpty = document.getElementById('result-empty');
const resultValue = document.getElementById('result-value');
const resultError = document.getElementById('result-error');

// Same feature names as the ML model
const FEATURES = [
  'Age',
  'Height',
  'Weight',
  'Duration',
  'Heart_Rate',
  'Body_Temp'
];

const LABELS = {
  Age: 'Age',
  Height: 'Height',
  Weight: 'Weight',
  Duration: 'Duration',
  Heart_Rate: 'Heart Rate',
  Body_Temp: 'Body Temp'
};

let meta = null;


// ---- Load model information ----
async function loadMeta() {
  try {
    const response = await fetch('/api/meta');

    if (!response.ok) {
      throw new Error('Failed to load model information.');
    }

    meta = await response.json();

  } catch (e) {
    showError(
      'Could not reach the prediction server. Make sure app.py is running.'
    );
  }
}


// ---- Validation ----
function validate() {
  let valid = true;

  FEATURES.forEach(name => {
    const input = document.getElementById(name);
    const errorEl = document.querySelector(`[data-error="${name}"]`);
    const field = input.closest('.field');

    let message = '';

    // Check only that the field is filled
    if (input.value.trim() === '') {
      message = `${LABELS[name]} is required.`;
    }

    // Check that the value is a valid number
    else if (Number.isNaN(Number(input.value))) {
      message = `${LABELS[name]} must be a valid number.`;
    }

    errorEl.textContent = message;
    field.classList.toggle('invalid', message !== '');

    if (message) {
      valid = false;
    }
  });

  return valid;
}


// ---- Show prediction result ----
function showResult(calories) {
  document.getElementById('result-kcal').textContent =
    calories.toFixed(1);

  const note = document.getElementById('result-note');

  note.textContent = meta
    ? `On unseen test data the model's average error was about ${meta.mae.toFixed(1)} kcal.`
    : '';

  resultEmpty.hidden = true;
  resultError.hidden = true;
  resultValue.hidden = false;
}


// ---- Show error ----
function showError(message) {
  document.getElementById('result-error-text').textContent = message;

  resultEmpty.hidden = true;
  resultValue.hidden = true;
  resultError.hidden = false;
}


// ---- Loading state ----
function setLoading(isLoading) {
  submitBtn.disabled = isLoading;

  submitBtn.querySelector('span').textContent =
    isLoading ? 'Predicting...' : 'Predict Calories';

  const icon = submitBtn.querySelector('svg');

  if (icon) {
    icon.classList.toggle('spin', isLoading);
  }
}


// ---- Submit form ----
form.addEventListener('submit', async (event) => {
  event.preventDefault();

  if (!validate()) {
    return;
  }

  const data = {
    Gender: form.elements['Gender'].value
  };

  FEATURES.forEach(name => {
    data[name] = Number(
      document.getElementById(name).value
    );
  });

  setLoading(true);

  try {
    const response = await fetch('/api/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(data)
    });

    const result = await response.json();

    if (!response.ok) {
      showError(result.error || 'Something went wrong.');
    } else {
      showResult(result.calories);

      // On small screens, bring the result into view
      if (window.innerWidth <= 960) {
        resultValue.scrollIntoView({
          behavior: 'smooth',
          block: 'center'
        });
      }
    }

  } catch (e) {
    showError(
      'Could not reach the prediction server. Make sure app.py is running.'
    );

  } finally {
    setLoading(false);
  }
});


// Load model information when the page starts
loadMeta();