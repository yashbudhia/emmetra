let zoomLevel = 1;

// Function to upload the selected image
async function uploadImage() {
  const fileInput = document.getElementById("file");
  if (!fileInput.files.length) {
    console.error("No file selected");
    return;
  }

  const formData = new FormData();
  formData.append("file", fileInput.files[0]);

  const response = await fetch("/upload", { method: "POST", body: formData });
  if (response.ok) {
    document.getElementById("processBtn").disabled = false;
    console.log("Image uploaded successfully");
  } else {
    console.error("Failed to upload image");
  }
}

// Function to update the processed image based on user inputs
async function updateImage() {
  const params = {
    width: parseInt(document.getElementById("width").value),
    height: parseInt(document.getElementById("height").value),
    gamma: parseFloat(document.getElementById("gamma").value),
    compression_factor: parseFloat(
      document.getElementById("compression").value
    ),
    white_balance: parseFloat(document.getElementById("white_balance").value),
    denoise: parseFloat(document.getElementById("denoise").value),
    sharpen: parseFloat(document.getElementById("sharpen").value),
    gaussian_blur: parseFloat(document.getElementById("gaussian_blur").value),
  };

  const response = await fetch("/process", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });

  if (response.ok) {
    const blob = await response.blob();
    const imgElement = document.getElementById("processed-image");
    imgElement.src = URL.createObjectURL(blob);
    imgElement.style.display = "block";
  } else {
    console.error("Image processing failed.");
  }
}

// Function to zoom the processed image
function zoomImage(direction) {
  zoomLevel += direction === "in" ? 0.1 : -0.1;
  document.getElementById(
    "processed-image"
  ).style.transform = `scale(${zoomLevel})`;
}

// Function to export the processed image
function exportImage() {
  const image = document.getElementById("processed-image");
  const link = document.createElement("a");
  link.href = image.src;
  link.download = "processed_image.jpg";
  link.click();
}

// Event listeners
document.getElementById("file").addEventListener("change", uploadImage);
document.getElementById("processBtn").addEventListener("click", updateImage);
document
  .getElementById("zoomIn")
  .addEventListener("click", () => zoomImage("in"));
document
  .getElementById("zoomOut")
  .addEventListener("click", () => zoomImage("out"));
document.getElementById("exportBtn").addEventListener("click", exportImage);
