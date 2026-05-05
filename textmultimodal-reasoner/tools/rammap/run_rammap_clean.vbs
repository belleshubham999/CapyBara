Set objShell = WScript.CreateObject("WScript.Shell")

' Change the path below to where your rammap.exe is located
strRamMap = "C:\Windows\rammap.exe"

' Run all 4 commands completely hidden
' The "0" at the end hides the window. "True" forces it to wait for the previous command to finish.
objShell.Run """" & strRamMap & """ -Em", 0, True
objShell.Run """" & strRamMap & """ -Es", 0, True
objShell.Run """" & strRamMap & """ -Et", 0, True
objShell.Run """" & strRamMap & """ -Ew", 0, True
