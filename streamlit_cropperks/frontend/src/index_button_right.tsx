import Cropper from "cropperjs"
import { RenderData, Streamlit } from "streamlit-component-lib"

// Create a main container with a flex row layout.
const mainContainer = document.body.appendChild(document.createElement("div"));
mainContainer.style.display = "flex";
mainContainer.style.flexDirection = "row";
mainContainer.style.alignItems = "center";
mainContainer.style.justifyContent = "center";
mainContainer.style.width = "100%";

// Create an image container.
const imageDiv = mainContainer.appendChild(document.createElement("div"));
imageDiv.style.flex = "1";
imageDiv.style.textAlign = "center";

// Create a button container.
const btnContainer = mainContainer.appendChild(document.createElement("div"));
btnContainer.style.marginLeft = "2rem";  // some horizontal spacing
btnContainer.style.display = "flex";
btnContainer.style.flexDirection = "column";
btnContainer.style.alignItems = "center";
btnContainer.style.justifyContent = "center";

// Create the image element inside imageDiv.
const img = imageDiv.appendChild(document.createElement("img"));
img.style.maxWidth = "100%";

// Create the crop button inside btnContainer.
const button = btnContainer.appendChild(document.createElement("button"));

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
  // Create an object URL for the image and assign it to the img element.
  img.src = URL.createObjectURL(new Blob([arrayBufferView], { type: "image/png" }));
  imageDiv.style.maxWidth = "100%";
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
        // When cropper is ready, wire the button to trigger cropping.
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