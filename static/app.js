// Get the elements for navigation and sections
const howToUseLink = document.getElementById("how-to-use");
const contactLink = document.getElementById("contact");
const howToUseSection = document.getElementById("how-to-use-section");
const contactSection = document.getElementById("contact-section");

// Add event listeners to scroll to sections
howToUseLink.addEventListener("click", (e) => {
  e.preventDefault();
  howToUseSection.scrollIntoView({
    behavior: "smooth",
    block: "start",
  });
});

contactLink.addEventListener("click", (e) => {
  e.preventDefault();
  contactSection.scrollIntoView({
    behavior: "smooth",
    block: "start",
  });
});

// Toggle menu on mobile
const menuToggle = document.querySelector(".menu-toggle");
const navbarLinks = document.querySelector(".navbar-links");

menuToggle.addEventListener("click", () => {
  navbarLinks.classList.toggle("mobile");
  menuToggle.classList.toggle("open");
});
