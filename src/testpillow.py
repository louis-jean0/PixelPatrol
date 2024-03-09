from PIL import Image
import customtkinter
im = Image.open("../data/lena.pgm")
print(im.format, im.size, im.mode)
#im.show()
window = customtkinter.CTk()
window.geometry("1920x1080")
window.title("Test")

def button_function():
    print("button pressed")

button = customtkinter.CTkButton(window,10,button_function)
button.place((0.5,0.5),customtkinter.CENTER)

customtkinter.set_appearance_mode("Dark")

window.mainloop()