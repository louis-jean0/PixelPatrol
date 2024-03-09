import customtkinter as ctk

app  = ctk.CTk()
app.geometry("400x300")
app.title("Test CustomTKinter")

button = ctk.CTkButton(master=app,text="Click me")
button.place(relx=0.5,rely=0.5,anchor=ctk.CENTER)

ctk.set_appearance_mode("Dark")

app.mainloop()