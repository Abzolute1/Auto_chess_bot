# Auto Chess Bot 🎮♟️

A chess bot for Chess.com and Lichess.org using Stockfish engine. Features human-like gameplay with variable thinking times, multiple play modes (auto/manual/mouseless), continuous puzzle solving, and move visualization. For educational purposes and unrated games only.

![GUI Screenshot](https://github.com/user-attachments/assets/6ee9bfa2-4440-455d-90fc-fb027aecaffe)



**This is a work in progress and updates will come**
![Animation_preview](https://github.com/user-attachments/assets/1c8332d9-b460-48cd-99f5-332af7d3b7a6)



##  **Features**


###  Human-like Behavior
- Variable thinking times based on position complexity
- Strategic decision-making that occasionally makes non-optimal moves
- Different response times for different move types (captures, checks, etc.)

###  Multiple Playing Modes
- Automatic mode for fully automated play
- Manual mode where the bot suggests moves but you make them
- Mouseless mode for Lichess.org (no mouse movements required)

###  Continuous Play Options
- Non-stop puzzle solving
- Non-stop online unrated games (**UNRATED ONLY!**)

###  Fun Features
- "Bongcloud" opening mode (the famous e4 followed by Ke2)
- Move history display and PGN export

##  Requirements

- Python 3.8 or higher
- Chrome browser (required - only works with Chrome)
- Stockfish chess engine executable

##  Dependencies

- selenium
- undetected-chromedriver
- webdriver-manager
- pyautogui
- python-chess
- pyqt6
- keyboard

##  Installation

1. Clone the repository:
git clone https://github.com/Aelexi93/Auto_chess_bot.git
cd Auto_chess_bot

2. Install required packages:


3. Download Chrome (required - only works with Chrome)

4. Download Stockfish:
- Get the latest version from [the official website](https://stockfishchess.org/download/)
- Extract the ZIP file to a location on your computer
- You'll select this location in the app's "Select Stockfish" button

##  Usage

1. Run the application: python src/gui.py

2. Click "Open Browser" to launch Chrome **"Must use the browser from the GUI and have Chrome installed!"**

3. Navigate to Chess.com or Lichess.org and select the appropriate option in the app

4. Configure your desired settings:
- Adjust Stockfish parameters (skill level, depth, memory, etc.)
- Enable/disable human-like behavior
- Set mouse latency for more natural movement

5. Click "Start" to begin playing

### Manual Mode
When manual mode is enabled, the bot will suggest moves (shown with an arrow overlay), but you'll need to press '3' to execute each move.

### Mouseless Mode
This mode is only available for Lichess.org and allows the bot to make moves without mouse movements, reducing the chance of detection.

## ⚠️ Ethical Considerations

This bot is intended for:
- Educational purposes
- Playing against other engines
- Testing chess strategies
- Playing unrated games

Please use responsibly and in accordance with the terms of service of chess websites.

##  Disclaimer

This project is for educational purposes only. Using bots for rated games on chess platforms violates their terms of service and can result in account bans. The developers of this tool are not responsible for any misuse or consequences thereof.

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
