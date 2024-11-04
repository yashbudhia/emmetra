let zoomLevel = 1;

document
  .querySelectorAll(
    "#gamma, #blur, #white_balance, #denoise, #sharpen, #demosaic"
  )
  .forEach((element) => element.addEventListener("input", updateImage));

async function uploadImage() {
  const fileInput = document.getElementById("file");
  const formData = new FormData();
  formData.append("file", fileInput.files[0]);

  await fetch("/upload", { method: "POST", body: formData });
  updateImage();
}

async function updateImage() {
  const formData = new FormData();
  formData.append("gamma", document.getElementById("gamma").value);
  formData.append(
    "white_balance",
    document.getElementById("white_balance").checked
  );
  formData.append("denoise", document.getElementById("denoise").checked);
  formData.append("sharpen", document.getElementById("sharpen").checked);
  formData.append("demosaic", document.getElementById("demosaic").checked);
  formData.append("blur", document.getElementById("blur").value);

  const response = await fetch("/process-image", {
    method: "POST",
    body: formData,
  });
  const blob = await response.blob();
  document.getElementById("processed-image").src = URL.createObjectURL(blob);
}

function zoomImage(direction) {
  zoomLevel += direction === "in" ? 0.1 : -0.1;
  document.getElementById(
    "processed-image"
  ).style.transform = `scale(${zoomLevel})`;
}

function exportImage() {
  const image = document.getElementById("processed-image");
  const link = document.createElement("a");
  link.href = image.src;
  link.download = "processed_image.jpg";
  link.click();
}
