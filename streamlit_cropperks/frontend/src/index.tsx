import Cropper from "cropperjs"
import { RenderData, Streamlit } from "streamlit-component-lib"

// Create a main container with a flex column layout.
const mainContainer = document.body.appendChild(document.createElement("div"));
mainContainer.style.display = "flex";
mainContainer.style.flexDirection = "column";
mainContainer.style.alignItems = "center";
mainContainer.style.width = "100%";

// Create an image container.
const imageDiv = mainContainer.appendChild(document.createElement("div"));
imageDiv.style.width = "100%";
imageDiv.style.textAlign = "center";

// Create a button container.
const btnDiv = mainContainer.appendChild(document.createElement("div"));
btnDiv.style.marginTop = "1rem";
btnDiv.style.textAlign = "center";

// Create the image element inside imageDiv.
const img = imageDiv.appendChild(document.createElement("img"));

// Create the crop button inside btnDiv.
const button = btnDiv.appendChild(document.createElement("button"));

// Add Cropper.js stylesheet.
const cropperStyle = document.head.appendChild(document.createElement("link"));
cropperStyle.rel = "stylesheet";
cropperStyle.href = "https://cdnjs.cloudflare.com/ajax/libs/cropperjs/1.5.13/cropper.min.css";

// Add custom style for our button (green with white text).
const customStyle = document.head.appendChild(document.createElement("style"));
customStyle.innerHTML = `
button {
  border-radius: 0.5rem;
  border: 1px solid rgba(49, 51, 63, 0.2);
  padding: 0.5rem 1rem;
  background-color: green;
  color: white;
  font-family: sans-serif;
  font-size: 16px;
  cursor: pointer;
}
button:focus {
  outline: none;
  border: 1px solid #004400;
}
`;

/**
 * The component's render function.
 */
function onRender(event: Event): void {
  const data = (event as CustomEvent<RenderData>).detail;

  // Set the button text (or default to "Crop Image")
  button.textContent = data.args["btn_text"] || "Crop Image";
  button.disabled = data.disabled;

  // Get the picture (as a Uint8Array) from the arguments.
  let pic = data.args["pic"];
  let arrayBufferView = new Uint8Array(pic);
  // Create an object URL for the image and assign to the img element.
  img.src = URL.createObjectURL(new Blob([arrayBufferView], { type: "image/png" }));
  imageDiv.style.maxWidth = "100%";
  img.style.maxWidth = "100%";
  img.id = data.args["key"];

  img.onload = function () {
    var cropper = new Cropper(img, {
      autoCropArea: 0.5,
      viewMode: 2,
      center: true,
      guides: false,
      rotatable: false,
      minContainerHeight:
        (imageDiv.clientWidth / img.naturalWidth) *
        img.naturalHeight *
        data.args["size"],
      ready: function () {
        // When the cropper is ready, add an event listener to our button.
        button.addEventListener("click", function () {
          var croppedImage = cropper.getCroppedCanvas().toDataURL("image/png");
          Streamlit.setComponentValue(croppedImage);
        });
        Streamlit.setFrameHeight();
      },
    });
  }
}

// Attach our render handler.
Streamlit.events.addEventListener(Streamlit.RENDER_EVENT, onRender);
Streamlit.setComponentReady();
Streamlit.setFrameHeight();