const body = document.querySelector('body'),
    sidebar = body.querySelector('nav'),
    toggle = body.querySelector(".toggle"),
    searchBtn = body.querySelector(".search-box"),
    modeSwitch = body.querySelector(".toggle-switch"),
    modeText = body.querySelector(".mode-text"),
    prevBtn = document.getElementById('prevBtn'),
    nextBtn = document.getElementById('nextBtn');

if (toggle && sidebar) {
    toggle.addEventListener("click", () => {
        sidebar.classList.toggle("close");
    });
}

if (searchBtn && sidebar) {
    searchBtn.addEventListener("click", () => {
        sidebar.classList.remove("close");
    });
}

if (modeSwitch && modeText) {
    modeSwitch.addEventListener("click", () => {
        body.classList.toggle("dark");
        if (body.classList.contains("dark")) {
            modeText.innerText = "Light mode";
        } else {
            modeText.innerText = "Dark mode";
        }
    });
}

if (prevBtn) {
    prevBtn.addEventListener('click', function () {
        const container = document.querySelector('.gallery-container');
        if (container) container.scrollLeft -= 300;
    });
}

if (nextBtn) {
    nextBtn.addEventListener('click', function () {
        const container = document.querySelector('.gallery-container');
        if (container) container.scrollLeft += 300;
    });
}