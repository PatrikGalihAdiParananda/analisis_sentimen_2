document.addEventListener("DOMContentLoaded", function() {
    // --- Variabel untuk elemen-elemen penting ---
    const hamburger = document.querySelector(".hamburger-btn");
    const sidebar = document.querySelector(".side-bar");
    const contents = document.querySelector(".contents-wrapper");
    const modeSwitch = document.querySelector(".mode-switch");
    const body = document.body;

    // --- Fungsi untuk menerapkan status sidebar dari localStorage ---
    const applySidebarState = () => {
        // Cek apakah status 'sidebar_closed' adalah 'true' di localStorage
        const isClosed = localStorage.getItem("sidebar_closed") === "true";
        
        // Terapkan class 'close' pada sidebar dan 'full-width' pada konten
        sidebar.classList.toggle("close", isClosed);
        if (contents) {
            contents.classList.toggle("full-width", isClosed);
        }
    };

    // --- Fungsi untuk menerapkan tema (gelap/terang) dari localStorage ---
    const applyTheme = () => {
        const theme = localStorage.getItem("theme");
        
        // Default ke dark-mode jika tidak ada tema tersimpan
        if (theme === "light") {
            body.classList.add("light-theme"); // Gunakan light-theme sesuai CSS baru
            modeSwitch.innerHTML = '<i class="fa-solid fa-moon"></i>';
        } else {
            body.classList.remove("light-theme");
            modeSwitch.innerHTML = '<i class="fa-solid fa-sun"></i>';
        }
    };
    
    // --- Terapkan state awal saat halaman pertama kali dimuat ---
    applySidebarState();
    applyTheme();

    // --- Event Listener untuk Tombol Hamburger (Sidebar Toggle) ---
    if (hamburger && sidebar) {
        hamburger.addEventListener("click", () => {
            const isClosed = sidebar.classList.toggle("close");
            if (contents) {
                contents.classList.toggle("full-width", isClosed);
            }
            // Simpan status sidebar ke localStorage
            localStorage.setItem("sidebar_closed", isClosed); 
        });
    }

    // --- Event Listener untuk Tombol Dark/Light Mode ---
    if (modeSwitch) {
        modeSwitch.addEventListener("click", () => {
            // Toggle class 'light-theme' pada body
            const isLightTheme = body.classList.toggle("light-theme");
            // Simpan pilihan tema ke localStorage
            localStorage.setItem("theme", isLightTheme ? "light" : "dark"); 
            // Panggil kembali fungsi applyTheme untuk update ikon
            applyTheme(); 
        });
    }
});