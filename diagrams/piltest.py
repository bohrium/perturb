from pil import image, imagedraw, imagefont

im = image.open("spacetime.png")

fnt = imagefont.truetype('pillow/tests/fonts/dejavusans.ttf', 18)

draw = imagedraw.draw(im)
draw.text((10,60), "world", font=fnt, fill=(0,0,0,255))
del draw

im.save('hi', "png")
