# Ηope

## Guidelines: Using Git
[//]: <> ( - **Always** use `git pull --rebase --autostash`)
- **Do not** use command line to stage changes. Use a visual tool (like Visual Studio Code or Git Gui) to preview and **confirm** the changes before commiting. A better way is to selectively stage only the necessary line changes.
- Each developed feature **should** have its own branch and each person should work on his own branch. After code review and validation, commits will be pushed to each **team branch** (AUEB and Phasmatic team branches). Since the are two developing teams, focusing on different aspects of the project at any given time, there should also be two different team branches.
- **Main** branch **SHOULD ONLY** hold the production **stable** version of the project.

## Guidelines: Working with a team
- **Do not** alter functions that were written by *another* contributor. Ask the contributor to change or let you change them, in order for the function to provide the necessary functionality.
- When altering existing functions, always check if the new behaviour affects other functions.

## Architecture

Server 
Client

### Server Features
- [ ] Data polling from intel SDK
- [ ] Object detection (bbox)
- [ ] Gesture detection
- [ ] Point cloud compression
- [ ] Server HTTP
- [ ] Server WebSockets
- [ ] Point cloud transmission

### Client Features
- [x] Connect with server
- [x] Point cloud decompression
- [x] 3D visualization
- [x] Virtual Reality



## How to Install

Before you apply the following steps.....
