# Records docs/demo.gif:  vhs docs/demo.tape   (run from the repo root, model already in the HF cache)
Output docs/demo.gif
Set Shell "bash"
Set FontSize 17
Set Width 1180
Set Height 380
Set Padding 26
Set LineHeight 1.35
Set TypingSpeed 45ms
Set Theme { "name": "laya-forge", "background": "#0D1117", "foreground": "#DCE3EC", "cursor": "#E8BE5E", "selection": "#2D3642", "black": "#0D1117", "brightBlack": "#5B6675", "red": "#E0A06A", "brightRed": "#E0A06A", "green": "#86ABE0", "brightGreen": "#86ABE0", "yellow": "#E8BE5E", "brightYellow": "#E8BE5E", "blue": "#86ABE0", "brightBlue": "#86ABE0", "magenta": "#BC96DA", "brightMagenta": "#BC96DA", "cyan": "#9AA6B5", "brightCyan": "#9AA6B5", "white": "#DCE3EC", "brightWhite": "#FFFFFF" }

Hide
Type "export PYTHONPATH=. HF_HUB_DISABLE_PROGRESS_BARS=1 PS1='$ ' && clear"
Enter
Show

Sleep 600ms
Type "python examples/demo.py --pace 0.9"
Sleep 400ms
Enter
Sleep 11s
