
const HOSTED_URLS = {
  model:
      'model_js/model.json',
  metadata:
      'model_js/metadata.json'
};

function settextField(text, predict) {
  const textField = document.getElementById('text-entry');
  textField.value = text;
  doPredict(predict);
}

function status(statusText) {
  console.log(statusText);
  document.getElementById('status').textContent = statusText;
}


function disableLoadModelButtons() {
  document.getElementById('load-model').style.display = 'none';
}

function doPredict(predict) {
  const textField = document.getElementById('text-entry');
  const result = predict(textField.value);
  
  var r = Math.round(result.score[0] * 255);
  var g = Math.round(result.score[1] * 255);
  var b = Math.round(result.score[2] * 255);
  
  var c = document.getElementById("myCanvas");
  var ctx = c.getContext("2d");
  ctx.clearRect(20, 20, 100, 100);
  ctx.fillStyle = "rgba(" + r + ", " + g + ", " + b + ", 1)";
  ctx.fillRect(20, 20, 100, 100);
  
  status("R, G, B: " + r + " " + g + " " + b + ' elapsed: ' + result.elapsed.toFixed(3) + ' ms)')
}

function prepUI(predict) {
  settextField("tensorflow orange", predict);
  const button2 = document.getElementById('predict-button');
  button2.addEventListener('click', () => doPredict(predict));
  button2.style.display = 'inline-block';
}

async function urlExists(url) {
  try {
    const response = await fetch(url, {method: 'HEAD'});
    return response.ok;
  } catch (err) {
    return false;
  }
}

async function loadHostedPretrainedModel(url) {
  try {
    const model = await tf.loadLayersModel(url);
    disableLoadModelButtons();
    return model;
  } catch (err) {
    console.error(err);
  }
}

async function loadHostedMetadata(url) {
  try {
    const metadataJson = await fetch(url);
    const metadata = await metadataJson.json();
    return metadata;
  } catch (err) {
    console.error(err);
  }
}

class Classifier {

  async init(urls) {
    this.urls = urls;
    this.model = await loadHostedPretrainedModel(urls.model);
    await this.loadMetadata();
    return this;
  }

  async loadMetadata() {
    const metadata =
        await loadHostedMetadata(this.urls.metadata);
    this.maxLen = metadata['max_len'];
    console.log('maxLen = ' + this.maxLen);
    this.wordIndex = metadata['word_index']
  }

  predict(text) {
    // Convert to lower case and remove all punctuations.
    const inputText =
        text.trim().toLowerCase().replace(/(\.|\,|\!)/g, '').split('');
    // Look up word indices.
    const inputBuffer = tf.buffer([1, this.maxLen], 'float32');
    for (let i = 0; i < inputText.length; ++i) {
      const word = inputText[i];
      inputBuffer.set(this.wordIndex[word], 0, i + this.maxLen - inputText.length);
      //console.log(word, this.wordIndex[word], inputBuffer);
    }
    const input = inputBuffer.toTensor();
    //console.log(input);

    const beginMs = performance.now();
    const predictOut = this.model.predict(input);
    //console.log(predictOut.dataSync());
    const values = predictOut.dataSync();//[0];
    predictOut.dispose();
    const endMs = performance.now();

    return {score: values, elapsed: (endMs - beginMs)};
  }
};

async function setup() {
  if (await urlExists(HOSTED_URLS.model)) {
    const button1 = document.getElementById('load-model');
    
    button1.addEventListener('click', async () => {
      const predictor = await new Classifier().init(HOSTED_URLS);
      prepUI(x => predictor.predict(x));
    });
    
    button1.style.display = 'inline-block';
  }
}

setup();
