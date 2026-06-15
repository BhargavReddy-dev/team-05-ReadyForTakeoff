import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QMessageBox, QTableWidgetItem
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor

from dashboard_ui import Ui_MainWindow
from delay_backend import score_last_day_schedule

class OCCApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.flight_records = []
        
        self.load_delay_risk_board()
        self.ui.flightTable.cellDoubleClicked.connect(self.evaluate_row_priority)
        self.ui.searchBar.textChanged.connect(self.filter_board)

    def load_delay_risk_board(self):
        try:
            scored_flights = score_last_day_schedule()
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Delay Model Load Failed",
                f"Unable to load the delay-risk backend.\n\n{exc}",
            )
            return

        self.flight_records = scored_flights.to_dict("records")

        table = self.ui.flightTable
        table.setSortingEnabled(False)
        table.clearContents()
        table.setRowCount(len(self.flight_records))
        table.setHorizontalHeaderLabels(
            [
                "Date",
                "Flight Number",
                "Tail Number",
                "Origin",
                "Destination",
                "Sched Dep",
                "Delay Risk",
                "Status",
            ]
        )

        for row_index, record in enumerate(self.flight_records):
            values = [
                self.format_date(record.get("sched_dep_dt")),
                self.format_flight_number(record),
                self.format_value(record.get("tail_number")),
                self.format_value(record.get("origin")),
                self.format_value(record.get("dest")),
                self.format_time(record.get("sched_dep_local")),
                self.format_probability(record.get("delay_probability")),
                self.format_status(record),
            ]

            row_color = self.risk_color(record.get("risk_band"))
            tooltip = self.detail_text(record)

            for column_index, value in enumerate(values):
                item = self.create_table_item(value, row_index, row_color, tooltip)
                table.setItem(row_index, column_index, item)

        table.resizeColumnsToContents()
        table.setSortingEnabled(True)

    def create_table_item(self, value, record_index, row_color, tooltip):
        item = QTableWidgetItem(value)
        item.setData(Qt.ItemDataRole.UserRole, record_index)
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        item.setToolTip(tooltip)

        if row_color is not None:
            item.setBackground(row_color)
            if row_color == QColor(255, 77, 77):
                item.setForeground(QColor("white"))
            else:
                item.setForeground(QColor("black"))

        return item

    def filter_board(self, text):
        query = text.strip().lower()
        table = self.ui.flightTable

        for row in range(table.rowCount()):
            row_text = []
            for column in range(table.columnCount()):
                item = table.item(row, column)
                if item:
                    row_text.append(item.text().lower())
            table.setRowHidden(row, query not in " ".join(row_text))

    def evaluate_row_priority(self, row, column):
        clicked_item = self.ui.flightTable.item(row, column)

        if not clicked_item:
            return

        record_index = clicked_item.data(Qt.ItemDataRole.UserRole)
        if record_index is None or record_index >= len(self.flight_records):
            return

        record = self.flight_records[record_index]
        risk = record.get("risk_band", "Unknown")
        message = self.detail_text(record)

        if risk == "High":
            QMessageBox.critical(self, "OCC High Priority Delay Alert", message)
        elif risk == "Medium":
            QMessageBox.warning(self, "OCC Moderate Priority Delay Warning", message)
        else:
            QMessageBox.information(self, "OCC Delay Risk Tracking", message)

    def detail_text(self, record):
        return (
            f"Flight: {self.format_flight_number(record)}\n"
            f"Route: {self.format_value(record.get('origin'))} to {self.format_value(record.get('dest'))}\n"
            f"Tail: {self.format_value(record.get('tail_number'))}\n"
            f"Scheduled departure: {self.format_datetime(record.get('sched_dep_dt'))}\n"
            f"Delay risk: {self.format_probability(record.get('delay_probability'))} "
            f"({record.get('risk_band', 'Unknown')})\n\n"
            f"Main reason: {record.get('main_reason', 'Unavailable')}\n\n"
            f"Explanation: {record.get('explanation', 'Unavailable')}\n\n"
            f"Recommended action: {record.get('recommended_action', 'Unavailable')}"
        )

    def risk_color(self, risk):
        if risk == "High":
            return QColor(255, 77, 77)
        if risk == "Medium":
            return QColor(255, 165, 0)
        if risk == "Low":
            return QColor(255, 255, 77)
        return None

    def format_flight_number(self, record):
        carrier = self.format_value(record.get("carrier"))
        flight_number = self.format_value(record.get("flight_number"))
        return f"{carrier}{flight_number}"

    def format_date(self, value):
        if hasattr(value, "strftime"):
            return value.strftime("%d %b")
        return self.format_value(value)

    def format_time(self, value):
        return self.format_value(value)

    def format_status(self, record):
        if float(record.get("cancelled") or 0) == 1:
            return f"Cancelled · {record.get('risk_band', 'Unknown')} risk"
        return f"{record.get('risk_band', 'Unknown')} risk"

    def format_datetime(self, value):
        if hasattr(value, "strftime"):
            return value.strftime("%Y-%m-%d %H:%M")
        return self.format_value(value)

    def format_probability(self, value):
        if self.is_missing(value):
            return "-"
        return f"{float(value) * 100:.0f}%"

    def format_minutes(self, value):
        if self.is_missing(value):
            return "-"
        return f"{int(round(float(value)))} min"

    def format_value(self, value):
        if self.is_missing(value):
            return "-"
        return str(value)

    def is_missing(self, value):
        return value is None or value != value

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = OCCApp()
    window.show()
    sys.exit(app.exec())
