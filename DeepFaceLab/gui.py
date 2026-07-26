import sys

from DeepFaceLab.shared.torch_config import configure_torch
configure_torch("gpu_train")


def main():
    try:
        from PyQt6.QtWidgets import QApplication
        from DeepFaceLab.gui_app.main_window import MainWindow
        from DeepFaceLab.gui_app.theme import apply_theme

        app = QApplication.instance() or QApplication(sys.argv)
        apply_theme(app)
        window = MainWindow()
        window.show()
        sys.exit(app.exec())

    except ImportError:
        print("PyQt6 is not installed. Please install it with: pip install PyQt6")
        print("You can use the CLI mode instead: python main.py <command>")
        sys.exit(1)


if __name__ == "__main__":
    main()
