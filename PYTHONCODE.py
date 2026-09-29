from tkinter import *
import random
from tkinter import messagebox

real_word = [
    "GOAT",
    "BOAT",
    "RAIN",
    "CODE",
    "DISH",
    "FISH",
    "WASH",
    "HAIR",
    "NOSE",
    "HAND"]

jumbled_word=[
    "AOGT",
    "TOAB",
    "INRA",
    "DECO",
    "DHIS",
    "SHFI",
    "SWAH",
    "IRHA",
    "SENO",
    "NDHA"]

options = random.randrange(0,len(jumbled_word),1)

def one():
    global jumbled_word, real_word,options
    lbl1.config(text=jumbled_word[options])
    
def answer():
    global jumbled_word,real_word,options
    options=random.randrange(0,len(jumbled_word),1)
    lbl1.config(text=jumbled_word[options])
    e1.delete(0,END)
   
def correct():
    global jumbled_word,real_word,options
    one_var=e1.get()
    if one_var==real_word[options]:
        messagebox.showinfo("congratulations!","your answer is correct.")
        answer()
    else:
        messagebox.showinfo("sorry!","Better luck next time.")
        e1.delete(0,END)
    

window=Tk()
window.geometry("350x400+400+300")
window.title("Jumbled Words Game")
window.configure(background = "yellow")

lbl1=Label(window,text="write here",font=("Arial",25,"bold"),bg="black",fg="white")
lbl1.pack(pady=30,ipady=10,ipadx=10)

correct_1=StringVar()
e1=Entry(window,font=("Arial",25,"bold"),textvariable=correct_1)
e1.pack(ipady=5,ipadx=5)

correct_button=Button(window,text="Check Out",font=("Arial",20,"bold"),width=20,bg="red",fg="white",relief=GROOVE,command=correct)
correct_button.pack(pady=40)

reset_button=Button(window,text="Reset",font=("Arial",20,"bold"),width=20,bg="blue",fg="white",relief=GROOVE,command=answer)
reset_button.pack()

one()
window.mainloop()