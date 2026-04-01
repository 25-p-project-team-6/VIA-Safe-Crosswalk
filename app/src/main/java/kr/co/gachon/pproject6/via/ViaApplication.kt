package kr.co.gachon.pproject6.via

import android.app.Application
import kr.co.gachon.pproject6.via.map.KineticGuestSessionManager

class ViaApplication : Application() {
    override fun onCreate() {
        super.onCreate()
        KineticGuestSessionManager.from(this).prefetchIfNeeded()
    }
}
