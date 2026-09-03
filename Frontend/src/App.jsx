import { useEffect, useState } from "react";
import "./App.css";

// Flask API URLs
const FEATURES_API = "http://127.0.0.1:5000/api/features";
const PREDICT_API = "http://127.0.0.1:5000/api/predict";

// Gemini chatbot works through Flask.
// DO NOT put Gemini API key here.
const CHAT_API = "http://127.0.0.1:5000/api/chat";

function App() {
  // ==========================================
  // FEATURES FROM FLASK
  // ==========================================

  const [features, setFeatures] = useState([]);
  const [loading, setLoading] = useState(true);

  // ==========================================
  // HOUSE INPUT VALUES
  // ==========================================

  const [inputs, setInputs] = useState({
    area: 125,
    bedrooms: 1,
    bathrooms: 2,
    mainroad: 1,
    basement: 0,
  });

  // ==========================================
  // PREDICTION STATE
  // ==========================================

  const [prediction, setPrediction] = useState(null);
  const [isPredicting, setIsPredicting] = useState(false);

  // Show / hide prediction form
  const [showForm, setShowForm] = useState(false);

  // ==========================================
  // AI ASSISTANT MESSAGES
  // ==========================================

  const [messages, setMessages] = useState([
    {
      type: "bot",
      text:
        "Hi! I’m your AI Assistant. I can answer questions about the Housing dataset, house prices, bedrooms, area, bathrooms, main road, basement, and ML predictions.\nHow can I help you today?",
    },
  ]);

  // ==========================================
  // AI CHAT LOADING
  // ==========================================

  const [isChatLoading, setIsChatLoading] = useState(false);

  // ==========================================
  // GET FEATURES FROM FLASK
  // ==========================================

  useEffect(() => {
    fetch(FEATURES_API)
      .then((response) => {
        if (!response.ok) {
          throw new Error(
            "Could not connect to Flask Features API"
          );
        }

        return response.json();
      })

      .then((data) => {
        console.log(
          "Features received from Flask:",
          data
        );

        if (Array.isArray(data)) {
          setFeatures(data);
        } else if (Array.isArray(data.features)) {
          setFeatures(data.features);
        } else {
          setFeatures(Object.keys(data));
        }

        setLoading(false);
      })

      .catch((error) => {
        console.error(
          "Features API Error:",
          error
        );

        setLoading(false);
      });
  }, []);

  // ==========================================
  // FORMAT PRICE IN INDIAN FORMAT
  // ==========================================

  const formatPrice = (value) => {
    if (
      value === null ||
      value === undefined ||
      Number.isNaN(Number(value))
    ) {
      return "--";
    }

    return new Intl.NumberFormat("en-IN").format(
      Number(value)
    );
  };

  // ==========================================
  // PRICE IN LAKHS
  // ==========================================

  const priceInLakhs =
    prediction !== null
      ? (Number(prediction) / 100000).toFixed(2)
      : "0.00";

  // ==========================================
  // HANDLE INPUT CHANGES
  // ==========================================

  const handleChange = (e) => {
    const { name, value } = e.target;

    setInputs((previousInputs) => ({
      ...previousInputs,
      [name]: Number(value),
    }));
  };

  // ==========================================
  // PREDICT HOUSE PRICE
  // ==========================================

  const predictHouse = async () => {
    try {
      setIsPredicting(true);

      // Current values entered by user
      const inputData = {
        area: Number(inputs.area),
        bedrooms: Number(inputs.bedrooms),
        bathrooms: Number(inputs.bathrooms),
        mainroad: Number(inputs.mainroad),
        basement: Number(inputs.basement),
      };

      console.log(
        "Sending input to Flask:",
        inputData
      );

      // Send data to Flask
      const response = await fetch(
        PREDICT_API,
        {
          method: "POST",

          headers: {
            "Content-Type": "application/json",
          },

          body: JSON.stringify(inputData),
        }
      );

      // Convert Flask response to JSON
      const data = await response.json();

      console.log(
        "Prediction received from Flask:",
        data
      );

      // Check API response
      if (!response.ok) {
        throw new Error(
          data.error ||
            "Prediction API failed"
        );
      }

      // Flask can return any of these names
      const predictedPrice =
        data.prediction ??
        data.predicted_price ??
        data.price;

      // Validate prediction
      if (
        predictedPrice === undefined ||
        predictedPrice === null ||
        Number.isNaN(
          Number(predictedPrice)
        )
      ) {
        throw new Error(
          "Flask did not return a valid prediction."
        );
      }

      // Update estimated price
      setPrediction(
        Number(predictedPrice)
      );

      // Close prediction form
      setShowForm(false);

    } catch (error) {
      console.error(
        "Prediction Error:",
        error
      );

      alert(
        "Unable to get prediction from Flask.\n\n" +
          error.message
      );
    } finally {
      setIsPredicting(false);
    }
  };

  // ==========================================
  // NEW PREDICTION BUTTON
  // ==========================================

  const newPrediction = () => {
    setShowForm(true);
  };

  // ==========================================
  // CHAT MESSAGE
  // ==========================================

  const sendMessage = async () => {
    const input =
      document.getElementById(
        "chatInput"
      );

    if (!input) return;

    const question =
      input.value.trim();

    if (!question) return;

    // ========================================
    // SHOW USER QUESTION
    // ========================================

    setMessages(
      (previousMessages) => [
        ...previousMessages,

        {
          type: "user",
          text: question,
        },
      ]
    );

    // Clear input
    input.value = "";

    // Show loading
    setIsChatLoading(true);

    try {
      // ======================================
      // SEND QUESTION TO FLASK
      // ======================================

      const chatData = {
        question: question,

        // Current values from the prediction form
        inputs: {
          area: Number(inputs.area),
          bedrooms: Number(inputs.bedrooms),
          bathrooms: Number(inputs.bathrooms),
          mainroad: Number(inputs.mainroad),
          basement: Number(inputs.basement),
        },

        // Current ML prediction
        prediction:
          prediction !== null
            ? Number(prediction)
            : null,
      };

      console.log(
        "Sending chatbot request:",
        chatData
      );

      const response = await fetch(
        CHAT_API,
        {
          method: "POST",

          headers: {
            "Content-Type":
              "application/json",
          },

          body: JSON.stringify(
            chatData
          ),
        }
      );

      const data =
        await response.json();

      console.log(
        "Gemini chatbot response:",
        data
      );

      // ======================================
      // CHECK RESPONSE
      // ======================================

      if (!response.ok) {
        throw new Error(
          data.error ||
            data.message ||
            "Chatbot API failed"
        );
      }

      // ======================================
      // GET GEMINI ANSWER
      // ======================================

      const answer =
        data.answer ||
        data.response ||
        data.message;

      if (!answer) {
        throw new Error(
          "Gemini did not return an answer."
        );
      }

      // ======================================
      // DISPLAY GEMINI ANSWER
      // ======================================

      setMessages(
        (previousMessages) => [
          ...previousMessages,

          {
            type: "bot",
            text: answer,
          },
        ]
      );

      // ======================================
      // IF FLASK RETURNED A NEW ML
      // PREDICTION, UPDATE PRICE CARD
      // ======================================

      if (
        data.prediction !==
          undefined &&
        data.prediction !== null
      ) {
        setPrediction(
          Number(data.prediction)
        );
      }

      if (
        data.scenario_prediction !==
          undefined &&
        data.scenario_prediction !== null
      ) {
        setPrediction(
          Number(
            data.scenario_prediction
          )
        );
      }

    } catch (error) {
      console.error(
        "Gemini Chatbot Error:",
        error
      );

      setMessages(
        (previousMessages) => [
          ...previousMessages,

          {
            type: "bot",
            text:
              "Sorry, I could not connect to the AI Assistant. Please make sure your Flask backend and Gemini API are running correctly.",
          },
        ]
      );
    } finally {
      setIsChatLoading(false);
    }
  };

  // ==========================================
  // CHAT ENTER KEY
  // ==========================================

  const handleChatKeyDown = (e) => {
    if (
      e.key === "Enter" &&
      !isChatLoading
    ) {
      e.preventDefault();

      sendMessage();
    }
  };

  // ==========================================
  // MAIN UI
  // ==========================================

  return (
    <div className="app">

      {/* ======================================
          HEADER
      ====================================== */}

      <header className="header">

        <div className="brand">

          <div className="brand-icon">
            ⌂
          </div>

          <h1>
            Smart Real Estate Valuation
          </h1>

        </div>

        <button
          className="header-button"
          onClick={newPrediction}
        >
          ⟳ &nbsp; New Prediction
        </button>

      </header>


      {/* ======================================
          DASHBOARD
      ====================================== */}

      <main className="dashboard">


        {/* ====================================
            LEFT CARD
        ==================================== */}

        <section className="card input-card">

          <div className="card-title blue">

            <span className="title-icon">
              ▣
            </span>

            <h2>
              Your Input Details
            </h2>

          </div>


          <div className="details">


            {/* AREA */}

            <div className="detail-row">

              <span className="detail-icon green">
                ▧
              </span>

              <span>
                Area (in sq ft)
              </span>

              <strong>
                {inputs.area}
              </strong>

            </div>


            {/* BEDROOMS */}

            <div className="detail-row">

              <span className="detail-icon">
                ▰
              </span>

              <span>
                Bedrooms
              </span>

              <strong>
                {inputs.bedrooms}
              </strong>

            </div>


            {/* BATHROOMS */}

            <div className="detail-row">

              <span className="detail-icon">
                ♨
              </span>

              <span>
                Bathrooms
              </span>

              <strong>
                {inputs.bathrooms}
              </strong>

            </div>


            {/* MAIN ROAD */}

            <div className="detail-row">

              <span className="detail-icon">
                ▲
              </span>

              <span>
                Main Road
              </span>

              <strong className="yes">

                {inputs.mainroad
                  ? "Yes"
                  : "No"}

              </strong>

            </div>


            {/* BASEMENT */}

            <div className="detail-row">

              <span className="detail-icon">
                ▰
              </span>

              <span>
                Basement
              </span>

              <strong className="yes">

                {inputs.basement
                  ? "Yes"
                  : "No"}

              </strong>

            </div>

          </div>


          {/* SUCCESS BOX */}

          <div className="success-box">

            <div className="success-icon">
              ✓
            </div>

            <div>

              <strong>
                All inputs look good!
              </strong>

              <p>
                Here is your predicted price.
              </p>

            </div>

          </div>


          {/* NEW PREDICTION */}

          <button
            className="main-button"
            onClick={newPrediction}
          >
            ▣ &nbsp; New Prediction
          </button>

        </section>


        {/* ====================================
            CENTER PRICE CARD
        ==================================== */}

        <section className="card prediction-card">


          <div className="card-title green-title">

            <span className="title-icon">
              ⌁
            </span>

            <h2>
              Estimated House Price
            </h2>

          </div>


          <div className="prediction-content">


            {/* PRICE */}

            <div className="price">

              {prediction !== null
                ? `₹ ${formatPrice(
                    prediction
                  )}`
                : "₹ --"}

            </div>


            {/* LAKHS */}

            <div className="lakhs">

              {prediction !== null
                ? `(${priceInLakhs} Lakhs)`
                : "(Waiting for prediction)"}

            </div>


            {/* STATUS */}

            <div className="completed">

              <span>
                ✓
              </span>

              {prediction !== null
                ? "Prediction completed successfully!"
                : "Enter details and predict"}

            </div>


            {/* HOUSE IMAGE */}

            <img
              className="house-image"
              src="https://images.unsplash.com/photo-1600585154340-be6161a56a0c?auto=format&fit=crop&w=1200&q=80"
              alt="Modern house"
            />


            {/* ABOUT */}

            <div className="about-box">

              <h3>
                ⓘ &nbsp; About this prediction
              </h3>

              <p>
                The predicted price is calculated
                using a Machine Learning model
                trained on historical house data.
                The prediction changes based on
                the house details you provide.
              </p>

            </div>


            {/* NOTE */}

            <div className="note-box">

              <span>
                💡
              </span>

              <p>

                <strong>
                  Note:
                </strong>{" "}
                The predicted price is an
                estimate and may vary in real life.

              </p>

            </div>

          </div>

        </section>


        {/* ====================================
            RIGHT AI ASSISTANT
        ==================================== */}

        <section className="card assistant-card">


          {/* ASSISTANT HEADER */}

          <div className="assistant-header">

            <div>

              <div className="assistant-title">

                <span className="robot">
                  ♟
                </span>

                <h2>
                  AI Assistant
                </h2>

              </div>

              <p>
                Ask me anything about this prediction!
              </p>

            </div>


            <button className="expand">
              ↗
            </button>

          </div>


          {/* CHAT AREA */}

          <div className="chat-area">

            {messages.map(
              (message, index) => (

                <div
                  key={index}
                  className={`message-row ${message.type}`}
                >


                  {/* BOT ICON */}

                  {message.type ===
                    "bot" && (

                    <div className="avatar bot-avatar">
                      ♟
                    </div>

                  )}


                  {/* MESSAGE */}

                  <div
                    className={`message ${message.type}`}
                  >

                    {message.text
                      .split("\n")
                      .map(
                        (line, i) => (

                          <span
                            key={i}
                          >
                            {line}
                            <br />
                          </span>

                        )
                      )}

                  </div>


                  {/* USER ICON */}

                  {message.type ===
                    "user" && (

                    <div className="avatar user-avatar">
                      ●
                    </div>

                  )}

                </div>

              )
            )}


            {/* ==================================
                GEMINI LOADING MESSAGE
            ================================== */}

            {isChatLoading && (

              <div className="message-row bot">

                <div className="avatar bot-avatar">
                  ♟
                </div>

                <div className="message bot">

                  Thinking...

                </div>

              </div>

            )}

          </div>


          {/* CHAT INPUT */}

          <div className="chat-input-area">

            <input
              id="chatInput"
              type="text"
              placeholder="Type your question..."
              onKeyDown={
                handleChatKeyDown
              }
              disabled={
                isChatLoading
              }
            />

            <button
              onClick={sendMessage}
              disabled={
                isChatLoading
              }
            >
              {isChatLoading
                ? "..."
                : "➤"}
            </button>

          </div>

        </section>

      </main>


      {/* ======================================
          FOOTER
      ====================================== */}

      <footer className="disclaimer">

        <span>
          ♢
        </span>

        <strong>
          Disclaimer:
        </strong>

        <p>
          This is an AI/ML based prediction.
          Actual property prices may vary
          based on market conditions, location,
          and other real-world factors.
        </p>

      </footer>


      {/* ======================================
          NEW PREDICTION MODAL
      ====================================== */}

      {showForm && (

        <div className="modal-overlay">

          <div className="modal">


            {/* MODAL HEADER */}

            <div className="modal-header">

              <h2>
                New House Price Prediction
              </h2>

              <button
                onClick={() =>
                  setShowForm(false)
                }
              >
                ×
              </button>

            </div>


            {/* AREA */}

            <div className="form-group">

              <label>
                Area (sq ft)
              </label>

              <input
                type="number"
                name="area"
                min="1"
                value={inputs.area}
                onChange={handleChange}
              />

            </div>


            {/* BEDROOMS */}

            <div className="form-group">

              <label>
                Bedrooms
              </label>

              <input
                type="number"
                name="bedrooms"
                min="1"
                value={inputs.bedrooms}
                onChange={handleChange}
              />

            </div>


            {/* BATHROOMS */}

            <div className="form-group">

              <label>
                Bathrooms
              </label>

              <input
                type="number"
                name="bathrooms"
                min="1"
                value={inputs.bathrooms}
                onChange={handleChange}
              />

            </div>


            {/* MAIN ROAD */}

            <div className="form-group">

              <label>
                Main Road
              </label>

              <select
                name="mainroad"
                value={inputs.mainroad}
                onChange={handleChange}
              >

                <option value="1">
                  Yes
                </option>

                <option value="0">
                  No
                </option>

              </select>

            </div>


            {/* BASEMENT */}

            <div className="form-group">

              <label>
                Basement
              </label>

              <select
                name="basement"
                value={inputs.basement}
                onChange={handleChange}
              >

                <option value="1">
                  Yes
                </option>

                <option value="0">
                  No
                </option>

              </select>

            </div>


            {/* PREDICT */}

            <button
              className="predict-button"
              onClick={predictHouse}
              disabled={isPredicting}
            >

              {isPredicting
                ? "Calculating Price..."
                : "Predict House Price"}

            </button>

          </div>

        </div>

      )}

    </div>
  );
}

export default App;